// io.S - console/file I/O, number formatting/parsing, timing, JSON escaping
// (owner: loader agent). No CRT: everything is built on kernel32.
//
// Exported (Microsoft x64 ABI, ints are 64-bit unless noted):
//   HANDLE io_stdout(void) / io_stderr(void) / io_stdin(void)
//   i64  io_write(HANDLE h, const void* buf, i64 len)   unbuffered; 0 ok, -1 error
//   void io_out(const void* buf, i64 len)   BUFFERED stdout (64 KB); call io_flush()
//   void io_out_z(const char* z)            io_out of a NUL-terminated string
//   void io_flush(void)                     write the stdout buffer out now
//   void io_err(const void* buf, i64 len)   unbuffered stderr
//   void io_err_z(const char* z)
//   u8*  io_read_stdin(i64* len)            the WHOLE of stdin (pipe, file or console)
//                                           in one mem_alloc'd buffer, NUL-terminated
//   u8*  io_read_file(const wchar_t* path, i64* len)  whole file (NUL-terminated), 0 on error
//   i64  io_crt_text(u8* buf, i64 len)      in place: what the C runtime's TEXT-mode
//                                           stdin would have delivered (see below); new len
//   i64  fmt_i64(char* buf, i64 v)          decimal, returns length (also NUL-terminates)
//   i64  fmt_u64(char* buf, u64 v)
//   i64  fmt_hex(char* buf, u64 v)          lowercase hex without 0x (like printf %x)
//   i64  fmt_f64(char* buf, double v, i64 decimals)   fixed-point like printf("%.*f")
//                                           (decimals 0..9; |v| < 9e18)
//   i64  parse_i64(const wchar_t* s)        atoi()-style: spaces, sign, digits
//   u64  parse_u64(const wchar_t* s)        strtoull(s, 0, 10)-style (overflow -> ~0)
//   double parse_f64(const wchar_t* s)      atof()-style decimal (exact for the usual
//                                           short inputs like 0.8, 1e-3 ...)
//   i64  io_time_ns(void)                   QueryPerformanceCounter in nanoseconds
//   i64  json_escape(u8* dst, const u8* src, i64 n)   escape like gen.c hf_emit_delta;
//                                           dst needs 6*n bytes; returns bytes written
//   i64  utf16_to_utf8(u8* dst, i64 cap, const wchar_t* src)  NUL-terminated result,
//                                           returns its length (0 on failure)
//
//   A tiny message builder for error/status lines (single-threaded use only):
//   void msg_z(const char*) / msg_n(const u8*, i64) / msg_w(const wchar_t*)
//        msg_i64(i64) / msg_u64(u64) / msg_hex(u64) / msg_f64(double, i64 decimals)
//        msg_nl()        appends "\r\n" (see "text mode" below)
//        msg_to_err()    writes the message to stderr and clears it
//        msg_to_out()    appends the message to the buffered stdout and clears it
//
// The C runtime's "text mode"
// ---------------------------
// The C engine reads stdin with fread() and writes with printf() WITHOUT setting
// binary mode, so on Windows the CRT translates: on input "\r\n" -> "\n" and a
// Ctrl-Z byte (0x1A) ends the input; on output "\n" -> "\r\n". To be a byte-exact
// drop-in we do the same by hand: io_crt_text() for input, and every line we print
// ends in "\r\n" (msg_nl, and the JSON lines in main.S).

    .intel_syntax noprefix
#include "model.inc"

#define STD_INPUT_HANDLE   -10
#define STD_OUTPUT_HANDLE  -11
#define STD_ERROR_HANDLE   -12
#define OBUF_SIZE          65536
#define MSG_SIZE           8192
#define GENERIC_READ       0x80000000
#define FILE_SHARE_READ    1
#define OPEN_EXISTING      3
#define FILE_ATTRIBUTE_NORMAL 0x80
#define FILE_FLAG_SEQUENTIAL_SCAN 0x08000000
#define CP_UTF8            65001

    .bss
    .balign 64
io_obuf:   .space OBUF_SIZE             // buffered stdout
io_olen:   .quad 0                      // bytes waiting in io_obuf
io_h_out:  .quad 0                      // cached std handles
io_h_err:  .quad 0
io_h_in:   .quad 0
io_qpf:    .quad 0                      // QueryPerformanceFrequency (ticks / second)
msg_len:   .quad 0
    .balign 64
msg_buf:   .space MSG_SIZE
msg_tmp:   .space 64                    // scratch for number formatting

    .section .rdata, "dr"
    .balign 8
crlf_str:  .ascii "\r\n"
hex_digits: .ascii "0123456789abcdef"
    .balign 8
// powers of ten as doubles, 1e0 .. 1e22 (all exactly representable)
pow10_tab:
    .double 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11
    .double 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22
// the same as integers, 10^0 .. 10^18
ipow10_tab:
    .quad 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000
    .quad 1000000000, 10000000000, 100000000000, 1000000000000
    .quad 10000000000000, 100000000000000, 1000000000000000
    .quad 10000000000000000, 100000000000000000, 1000000000000000000
f64_9e18:      .double 9.0e18
nan_str:   .asciz "nan"
huge_str:  .asciz "huge"

    .text

// ===========================================================================
// standard handles (fetched once, then cached)
// ===========================================================================
    .globl io_stdout
io_stdout:
    mov rax, qword ptr [rip + io_h_out]
    test rax, rax
    jnz 1f
    sub rsp, 40
    mov ecx, STD_OUTPUT_HANDLE
    call qword ptr [rip + __imp_GetStdHandle]
    add rsp, 40
    mov qword ptr [rip + io_h_out], rax
1:  ret

    .globl io_stderr
io_stderr:
    mov rax, qword ptr [rip + io_h_err]
    test rax, rax
    jnz 1f
    sub rsp, 40
    mov ecx, STD_ERROR_HANDLE
    call qword ptr [rip + __imp_GetStdHandle]
    add rsp, 40
    mov qword ptr [rip + io_h_err], rax
1:  ret

    .globl io_stdin
io_stdin:
    mov rax, qword ptr [rip + io_h_in]
    test rax, rax
    jnz 1f
    sub rsp, 40
    mov ecx, STD_INPUT_HANDLE
    call qword ptr [rip + __imp_GetStdHandle]
    add rsp, 40
    mov qword ptr [rip + io_h_in], rax
1:  ret

// ===========================================================================
// i64 io_write(HANDLE h, const void* buf, i64 len) -> 0 ok, -1 error
// WriteFile takes a 32-bit count, so big buffers go out in 1 GB pieces.
// ===========================================================================
    .globl io_write
io_write:
    push rbx
    push rsi
    push rdi
    sub rsp, 48                         // [rsp+32] = 5th arg, [rsp+40] = DWORD written
    mov rbx, rcx                        // rbx = handle
    mov rsi, rdx                        // rsi = cursor
    mov rdi, r8                         // rdi = bytes left
.Liw_loop:
    test rdi, rdi
    jle .Liw_ok
    mov r8, rdi
    cmp r8, 0x40000000
    jbe 1f
    mov r8d, 0x40000000
1:  mov rcx, rbx                        // WriteFile(h, buf, n, &written, NULL)
    mov rdx, rsi
    lea r9, [rsp + 40]
    mov dword ptr [rsp + 40], 0
    mov qword ptr [rsp + 32], 0
    call qword ptr [rip + __imp_WriteFile]
    test eax, eax
    jz .Liw_err
    mov eax, dword ptr [rsp + 40]       // zero-extends into rax
    test eax, eax
    jz .Liw_err
    add rsi, rax
    sub rdi, rax
    jmp .Liw_loop
.Liw_ok:
    xor eax, eax
    jmp .Liw_ret
.Liw_err:
    mov rax, -1
.Liw_ret:
    add rsp, 48
    pop rdi
    pop rsi
    pop rbx
    ret

// ===========================================================================
// buffered stdout
// ===========================================================================
    .globl io_flush
io_flush:
    sub rsp, 40
    mov r8, qword ptr [rip + io_olen]
    test r8, r8
    jz 1f
    call io_stdout
    mov rcx, rax
    lea rdx, [rip + io_obuf]
    mov r8, qword ptr [rip + io_olen]
    call io_write
    mov qword ptr [rip + io_olen], 0
1:  add rsp, 40
    ret

// void io_out(const void* buf, i64 len)
    .globl io_out
io_out:
    push rbx
    push rsi
    push rdi
    sub rsp, 32
    mov rsi, rcx                        // rsi = source
    mov rbx, rdx                        // rbx = length
    test rbx, rbx
    jle .Lio_ret
    mov rax, qword ptr [rip + io_olen]
    add rax, rbx
    cmp rax, OBUF_SIZE
    jbe .Lio_copy                       // fits behind what is buffered
    call io_flush                       // make room
    cmp rbx, OBUF_SIZE
    jb .Lio_copy
    call io_stdout                      // bigger than the whole buffer: write through
    mov rcx, rax
    mov rdx, rsi
    mov r8, rbx
    call io_write
    jmp .Lio_ret
.Lio_copy:
    lea rdi, [rip + io_obuf]
    add rdi, qword ptr [rip + io_olen]
    mov rcx, rbx
    rep movsb                           // copy rcx bytes [rsi] -> [rdi]
    add qword ptr [rip + io_olen], rbx
.Lio_ret:
    add rsp, 32
    pop rdi
    pop rsi
    pop rbx
    ret

// i64 z_len(const char* z): strlen (internal, clobbers rax rdx only)
z_len:
    mov rax, rcx
1:  cmp byte ptr [rax], 0
    je 2f
    inc rax
    jmp 1b
2:  sub rax, rcx
    ret

    .globl io_out_z
io_out_z:
    sub rsp, 40
    call z_len
    mov rdx, rax
    call io_out                         // rcx still = z
    add rsp, 40
    ret

// void io_err(const void* buf, i64 len): unbuffered stderr
    .globl io_err
io_err:
    push rbx
    push rsi
    sub rsp, 40
    mov rbx, rcx
    mov rsi, rdx
    call io_stderr
    mov rcx, rax
    mov rdx, rbx
    mov r8, rsi
    call io_write
    add rsp, 40
    pop rsi
    pop rbx
    ret

    .globl io_err_z
io_err_z:
    sub rsp, 40
    call z_len
    mov rdx, rax
    call io_err
    add rsp, 40
    ret

// ===========================================================================
// u8* io_read_stdin(i64* len)
// Reads until EOF (ReadFile returns 0 bytes, or fails with ERROR_BROKEN_PIPE when
// the writer of a pipe closes). The buffer doubles as needed; one byte is always
// kept free for the terminating NUL.
// ===========================================================================
    .globl io_read_stdin
io_read_stdin:
    push rbx                            // rbx = buffer
    push rsi                            // rsi = capacity
    push rdi                            // rdi = bytes read so far
    push r12                            // r12 = len out-pointer
    push r13                            // r13 = stdin handle
    sub rsp, 48                         // 8 + 5*8 + 48 = 96
    mov r12, rcx
    call io_stdin
    mov r13, rax
    mov esi, 65536
    mov rcx, rsi
    call mem_alloc
    mov rbx, rax
    xor edi, edi
    test rbx, rbx
    jz .Lrs_done
    test r13, r13                       // no stdin at all (0 or INVALID_HANDLE_VALUE)
    jz .Lrs_done
    cmp r13, -1
    je .Lrs_done
.Lrs_loop:
    lea rax, [rdi + 1]
    cmp rax, rsi
    jb .Lrs_read
    // full: grow to twice the size
    lea rcx, [rsi + rsi]
    call mem_alloc
    test rax, rax
    jz .Lrs_done                        // out of memory: return what we have
    push rax                            // (keeps alignment: two pushes)
    push rax
    mov rcx, rax
    mov rdx, rbx
    mov r8, rdi
    sub rsp, 32
    call mem_copy
    mov rcx, rbx
    call mem_free
    add rsp, 32
    pop rax
    pop rbx                             // rbx = new buffer
    add rsi, rsi
.Lrs_read:
    mov rcx, r13                        // ReadFile(h, buf+n, cap-n-1, &got, NULL)
    lea rdx, [rbx + rdi]
    mov r8, rsi
    sub r8, rdi
    dec r8
    cmp r8, 0x40000000
    jbe 1f
    mov r8d, 0x40000000
1:  lea r9, [rsp + 40]
    mov dword ptr [rsp + 40], 0
    mov qword ptr [rsp + 32], 0
    call qword ptr [rip + __imp_ReadFile]
    test eax, eax
    jz .Lrs_done                        // error / broken pipe = end of input
    mov eax, dword ptr [rsp + 40]
    test eax, eax
    jz .Lrs_done                        // 0 bytes = end of file
    add rdi, rax
    jmp .Lrs_loop
.Lrs_done:
    test rbx, rbx
    jz 1f
    mov byte ptr [rbx + rdi], 0
1:  test r12, r12
    jz 2f
    mov qword ptr [r12], rdi
2:  mov rax, rbx
    add rsp, 48
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

// ===========================================================================
// u8* io_read_file(const wchar_t* path, i64* len) -> NUL-terminated buffer or 0
// ===========================================================================
    .globl io_read_file
io_read_file:
    push rbx                            // handle
    push rsi                            // buffer
    push rdi                            // size
    push r12                            // len out-pointer
    push r13                            // bytes read
    sub rsp, 64                         // 8 + 40 + 64 = 112
    mov r12, rdx
    xor esi, esi
    // CreateFileW(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, flags, NULL)
    mov edx, GENERIC_READ
    mov r8d, FILE_SHARE_READ
    xor r9d, r9d
    mov qword ptr [rsp + 32], OPEN_EXISTING
    mov qword ptr [rsp + 40], FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN
    mov qword ptr [rsp + 48], 0
    call qword ptr [rip + __imp_CreateFileW]
    cmp rax, -1
    je .Lrf_ret
    mov rbx, rax
    mov rcx, rbx                        // GetFileSizeEx(h, &size)
    lea rdx, [rsp + 56]
    call qword ptr [rip + __imp_GetFileSizeEx]
    test eax, eax
    jz .Lrf_close
    mov rdi, qword ptr [rsp + 56]
    lea rcx, [rdi + 1]
    call mem_alloc
    mov rsi, rax
    test rsi, rsi
    jz .Lrf_close
    xor r13d, r13d
.Lrf_loop:
    cmp r13, rdi
    jae .Lrf_ok
    mov rcx, rbx
    lea rdx, [rsi + r13]
    mov r8, rdi
    sub r8, r13
    cmp r8, 0x40000000
    jbe 1f
    mov r8d, 0x40000000
1:  lea r9, [rsp + 48]
    mov dword ptr [rsp + 48], 0
    mov qword ptr [rsp + 32], 0
    call qword ptr [rip + __imp_ReadFile]
    test eax, eax
    jz .Lrf_fail
    mov eax, dword ptr [rsp + 48]
    test eax, eax
    jz .Lrf_fail                        // file shrank under us
    add r13, rax
    jmp .Lrf_loop
.Lrf_fail:
    mov rcx, rsi
    call mem_free
    xor esi, esi
    jmp .Lrf_close
.Lrf_ok:
    mov byte ptr [rsi + rdi], 0
    test r12, r12
    jz .Lrf_close
    mov qword ptr [r12], rdi
.Lrf_close:
    mov rcx, rbx
    call qword ptr [rip + __imp_CloseHandle]
.Lrf_ret:
    mov rax, rsi
    add rsp, 64
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

// ===========================================================================
// i64 io_crt_text(u8* buf, i64 len): CRT text-mode input translation, in place.
//   0x1A (Ctrl-Z) ends the input; "\r\n" becomes "\n"; a lone "\r" is kept.
// (Verified against a mingw-built fread: "a\r\nb\rc\r\r\nd\x1ae" -> "a\nb\rc\r\nd".)
// ===========================================================================
    .globl io_crt_text
io_crt_text:
    xor eax, eax                        // rax = read index
    xor r8d, r8d                        // r8  = write index
.Lct_loop:
    cmp rax, rdx
    jae .Lct_end
    movzx r9d, byte ptr [rcx + rax]
    cmp r9d, 0x1A
    je .Lct_end
    cmp r9d, 0x0D
    jne .Lct_put
    lea r10, [rax + 1]
    cmp r10, rdx
    jae .Lct_put
    cmp byte ptr [rcx + r10], 0x0A
    jne .Lct_put
    mov r9d, 0x0A                       // "\r\n" -> "\n"
    inc rax
.Lct_put:
    mov byte ptr [rcx + r8], r9b
    inc r8
    inc rax
    jmp .Lct_loop
.Lct_end:
    mov byte ptr [rcx + r8], 0
    mov rax, r8
    ret

// ===========================================================================
// number formatting
// ===========================================================================
// i64 fmt_u64(char* buf, u64 v): digits are produced last-first into a small
// scratch on the stack, then copied out in the right order.
    .globl fmt_u64
fmt_u64:
    sub rsp, 40                         // 32 bytes of digit scratch at [rsp]
    mov r8, rcx                         // r8 = destination
    mov rax, rdx                        // rax = value
    lea r9, [rsp + 32]                  // r9 = scratch cursor (fills downwards)
    mov r10d, 10
1:  xor edx, edx
    div r10                             // rdx:rax / 10 -> rax quotient, rdx remainder
    add dl, '0'
    dec r9
    mov byte ptr [r9], dl
    test rax, rax
    jnz 1b
    lea rax, [rsp + 32]
    sub rax, r9                         // rax = number of digits
    xor ecx, ecx
2:  mov dl, byte ptr [r9 + rcx]
    mov byte ptr [r8 + rcx], dl
    inc rcx
    cmp rcx, rax
    jb 2b
    mov byte ptr [r8 + rax], 0
    add rsp, 40
    ret

// i64 fmt_i64(char* buf, i64 v)
    .globl fmt_i64
fmt_i64:
    sub rsp, 40
    test rdx, rdx
    jns 1f
    mov byte ptr [rcx], '-'
    inc rcx
    neg rdx                             // |INT64_MIN| still prints right as unsigned
    call fmt_u64
    inc rax
    jmp 2f
1:  call fmt_u64
2:  add rsp, 40
    ret

// i64 fmt_hex(char* buf, u64 v): lowercase, no leading zeros (printf "%x")
    .globl fmt_hex
fmt_hex:
    sub rsp, 40
    mov r8, rcx
    lea r9, [rsp + 32]
    lea r10, [rip + hex_digits]
1:  mov rax, rdx
    and eax, 15
    mov al, byte ptr [r10 + rax]
    dec r9
    mov byte ptr [r9], al
    shr rdx, 4
    jnz 1b
    lea rax, [rsp + 32]
    sub rax, r9
    xor ecx, ecx
2:  mov dl, byte ptr [r9 + rcx]
    mov byte ptr [r8 + rcx], dl
    inc rcx
    cmp rcx, rax
    jb 2b
    mov byte ptr [r8 + rax], 0
    add rsp, 40
    ret

// i64 fmt_f64(char* buf, double v /* xmm1 */, i64 decimals /* r8 */)
// Fixed point: R = round_half_even(|v| * 10^d) as an integer, then print
// R / 10^d "." R % 10^d zero-padded to d digits. Good for timings and rates.
    .globl fmt_f64
fmt_f64:
    push rbx
    push rsi
    push rdi
    push r12
    sub rsp, 40                         // 8 + 32 + 40 = 80
    mov rbx, rcx                        // rbx = output cursor
    mov rsi, rcx                        // rsi = output start
    ucomisd xmm1, xmm1                  // NaN is unordered with itself
    jp .Lff_nan
    test r8, r8
    jns 1f
    xor r8d, r8d
1:  cmp r8, 9
    jle 2f
    mov r8d, 9
2:  mov r12, r8                         // r12 = decimals
    xorpd xmm0, xmm0
    ucomisd xmm1, xmm0
    jae 3f
    mov byte ptr [rbx], '-'             // negative: print the sign, use |v|
    inc rbx
    subsd xmm0, xmm1                    // xmm0 = 0 - v = |v|
    movapd xmm1, xmm0
3:  lea rax, [rip + pow10_tab]
    mulsd xmm1, qword ptr [rax + r12*8] // |v| * 10^d
    ucomisd xmm1, qword ptr [rip + f64_9e18]
    jae .Lff_huge
    cvtsd2si rax, xmm1                  // round to nearest (MXCSR default: even)
    lea rcx, [rip + ipow10_tab]
    mov rdi, qword ptr [rcx + r12*8]    // rdi = 10^d
    xor edx, edx
    div rdi                             // rax = integer part, rdx = fraction digits
    mov rdi, rdx                        // keep the fraction
    mov rcx, rbx
    mov rdx, rax
    call fmt_u64
    add rbx, rax
    test r12, r12
    jz .Lff_end
    mov byte ptr [rbx], '.'
    inc rbx
    // write the fraction as exactly r12 digits, last digit first
    mov rax, rdi
    mov rcx, r12
    mov r10d, 10
4:  xor edx, edx
    div r10
    add dl, '0'
    mov byte ptr [rbx + rcx - 1], dl
    dec rcx
    jnz 4b
    add rbx, r12
    jmp .Lff_end
.Lff_nan:
    lea rdx, [rip + nan_str]
    jmp .Lff_copy
.Lff_huge:
    lea rdx, [rip + huge_str]
.Lff_copy:
    mov al, byte ptr [rdx]
    test al, al
    jz .Lff_end
    mov byte ptr [rbx], al
    inc rbx
    inc rdx
    jmp .Lff_copy
.Lff_end:
    mov byte ptr [rbx], 0
    mov rax, rbx
    sub rax, rsi
    add rsp, 40
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

// ===========================================================================
// number parsing from UTF-16 command-line strings
// ===========================================================================
// skip_ws (internal): rcx = wchar_t*; advances rcx past C isspace() chars
// (space, \t \n \v \f \r). Clobbers eax.
skip_ws:
1:  movzx eax, word ptr [rcx]
    cmp eax, ' '
    je 2f
    cmp eax, 9
    jb 3f
    cmp eax, 13
    ja 3f
2:  add rcx, 2
    jmp 1b
3:  ret

// i64 parse_i64(const wchar_t* s): like atoi (no overflow check)
    .globl parse_i64
parse_i64:
    call skip_ws
    xor r8d, r8d                        // r8 = negative?
    movzx eax, word ptr [rcx]
    cmp eax, '-'
    jne 1f
    mov r8d, 1
    add rcx, 2
    jmp 2f
1:  cmp eax, '+'
    jne 2f
    add rcx, 2
2:  xor eax, eax                        // rax = value
3:  movzx edx, word ptr [rcx]
    sub edx, '0'
    cmp edx, 9
    ja 4f                               // unsigned compare: also catches chars < '0'
    imul rax, rax, 10
    add rax, rdx
    add rcx, 2
    jmp 3b
4:  test r8d, r8d
    jz 5f
    neg rax
5:  ret

// u64 parse_u64(const wchar_t* s): like strtoull(s, NULL, 10): overflow gives
// 0xFFFFFFFFFFFFFFFF, a leading '-' negates (mod 2^64)
    .globl parse_u64
parse_u64:
    call skip_ws
    xor r8d, r8d
    movzx eax, word ptr [rcx]
    cmp eax, '-'
    jne 1f
    mov r8d, 1
    add rcx, 2
    jmp 2f
1:  cmp eax, '+'
    jne 2f
    add rcx, 2
2:  xor eax, eax
    xor r9d, r9d                        // r9 = overflowed?
3:  movzx edx, word ptr [rcx]
    sub edx, '0'
    cmp edx, 9
    ja 4f
    mov r10d, 10
    push rdx
    mul r10                             // rdx:rax = rax * 10 (rdx = high half)
    test rdx, rdx
    pop rdx
    jnz 6f
    add rax, rdx
    jc 6f
    add rcx, 2
    jmp 3b
6:  mov r9d, 1                          // overflow: keep consuming digits
    add rcx, 2
    jmp 3b
4:  test r9d, r9d
    jz 5f
    mov rax, -1
    ret
5:  test r8d, r8d
    jz 7f
    neg rax
7:  ret

// double parse_f64(const wchar_t* s): like atof for plain decimals
//   [spaces] [+|-] digits [. digits] [(e|E) [+|-] digits]
// Up to 18 significant digits are kept in an integer mantissa M with a decimal
// exponent E (value = M * 10^E). When M < 2^53 and |E| <= 22, M and 10^|E| are both
// exact doubles, so ONE multiply/divide gives the correctly rounded result - the
// same double strtod returns (Clinger's fast path). Other inputs are scaled in
// steps (may differ from strtod in the last bit).
    .globl parse_f64
parse_f64:
    push rbx
    push rsi
    call skip_ws
    xor r8d, r8d                        // r8 = negative?
    movzx eax, word ptr [rcx]
    cmp eax, '-'
    jne 1f
    mov r8d, 1
    add rcx, 2
    jmp 2f
1:  cmp eax, '+'
    jne 2f
    add rcx, 2
2:  xor eax, eax                        // rax = mantissa M
    xor r9d, r9d                        // r9  = significant digits kept
    xor r10d, r10d                      // r10 = decimal exponent E
    xor r11d, r11d                      // r11 = seen '.'?
    xor ebx, ebx                        // rbx = any digit seen?
.Lpf_digits:
    movzx edx, word ptr [rcx]
    cmp edx, '.'
    jne 3f
    test r11d, r11d
    jnz .Lpf_mant_done                  // a second '.' ends the number
    mov r11d, 1
    add rcx, 2
    jmp .Lpf_digits
3:  sub edx, '0'
    cmp edx, 9
    ja .Lpf_mant_done
    mov ebx, 1
    add rcx, 2
    test rax, rax                       // leading zeros are not significant
    jnz 4f
    test edx, edx
    jnz 4f
    test r11d, r11d                     // a zero after the point still moves E
    jz .Lpf_digits
    dec r10
    jmp .Lpf_digits
4:  cmp r9, 18
    jae 5f
    imul rax, rax, 10
    add rax, rdx
    inc r9
    test r11d, r11d
    jz .Lpf_digits
    dec r10                             // a kept fraction digit: E -= 1
    jmp .Lpf_digits
5:  test r11d, r11d                     // digit beyond 18 significant: dropped;
    jnz .Lpf_digits                     // an integer-part one still scales by 10
    inc r10
    jmp .Lpf_digits
.Lpf_mant_done:
    test ebx, ebx
    jz .Lpf_zero                        // no digits at all: 0.0
    // optional exponent
    movzx edx, word ptr [rcx]
    or edx, 0x20                        // 'E' -> 'e'
    cmp edx, 'e'
    jne .Lpf_build
    lea rsi, [rcx + 2]
    xor r11d, r11d                      // r11 = exponent negative?
    movzx edx, word ptr [rsi]
    cmp edx, '-'
    jne 6f
    mov r11d, 1
    add rsi, 2
    jmp 7f
6:  cmp edx, '+'
    jne 7f
    add rsi, 2
7:  movzx edx, word ptr [rsi]           // need at least one digit, else no exponent
    sub edx, '0'
    cmp edx, 9
    ja .Lpf_build
    xor ebx, ebx                        // rbx = exponent value (capped)
8:  movzx edx, word ptr [rsi]
    sub edx, '0'
    cmp edx, 9
    ja 9f
    cmp rbx, 100000
    jae 10f
    imul rbx, rbx, 10
    add rbx, rdx
10: add rsi, 2
    jmp 8b
9:  test r11d, r11d
    jz 11f
    neg rbx
11: add r10, rbx
.Lpf_build:
    test rax, rax
    jz .Lpf_zero
    cvtsi2sd xmm0, rax                  // M < 10^18 < 2^63: signed convert is fine
    lea rdx, [rip + pow10_tab]
    test r10, r10
    js .Lpf_neg_exp
.Lpf_pos_exp:                           // multiply by 10^E, 22 at a time
    cmp r10, 22
    jle 12f
    mulsd xmm0, qword ptr [rdx + 22*8]
    sub r10, 22
    cmp r10, 400                        // already infinite
    jg 12f
    jmp .Lpf_pos_exp
12: cmp r10, 22
    jle 13f
    mov r10d, 22
13: mulsd xmm0, qword ptr [rdx + r10*8]
    jmp .Lpf_sign
.Lpf_neg_exp:
    neg r10
14: cmp r10, 22
    jle 15f
    divsd xmm0, qword ptr [rdx + 22*8]
    sub r10, 22
    cmp r10, 400
    jg .Lpf_zero
    jmp 14b
15: divsd xmm0, qword ptr [rdx + r10*8]
.Lpf_sign:
    test r8d, r8d
    jz 16f
    xorpd xmm1, xmm1
    subsd xmm1, xmm0
    movapd xmm0, xmm1
16: pop rsi
    pop rbx
    ret
.Lpf_zero:
    xorpd xmm0, xmm0
    jmp .Lpf_sign

// ===========================================================================
// i64 io_time_ns(void): QueryPerformanceCounter scaled to nanoseconds.
// ticks * 1e9 is formed as a 128-bit product (mul gives rdx:rax) and divided by
// the frequency, so nothing overflows.
// ===========================================================================
    .globl io_time_ns
io_time_ns:
    sub rsp, 40
    cmp qword ptr [rip + io_qpf], 0
    jne 1f
    lea rcx, [rip + io_qpf]
    call qword ptr [rip + __imp_QueryPerformanceFrequency]
1:  lea rcx, [rsp + 32]
    call qword ptr [rip + __imp_QueryPerformanceCounter]
    mov rax, qword ptr [rsp + 32]
    mov ecx, 1000000000
    mul rcx
    div qword ptr [rip + io_qpf]
    add rsp, 40
    ret

// ===========================================================================
// i64 json_escape(u8* dst, const u8* src, i64 n)
// Byte for byte what gen.c hf_emit_delta prints between {"d":" and "}:
//   "  -> \"      \  -> \\      LF -> \n     CR -> \r     TAB -> \t
//   any other byte < 0x20 -> \u00XX (lowercase hex, printf "\\u%04x")
//   everything else (incl. 0x7F and all UTF-8 bytes >= 0x80) is copied raw.
// ===========================================================================
    .globl json_escape
json_escape:
    mov r9, rcx                         // r9 = dst start
    lea r10, [rip + hex_digits]
    test r8, r8
    jle .Lje_end
    add r8, rdx                         // r8 = src end
.Lje_loop:
    movzx eax, byte ptr [rdx]
    inc rdx
    cmp eax, 0x20
    jb .Lje_ctl
    cmp eax, '"'
    je .Lje_bs
    cmp eax, '\\'
    je .Lje_bs
    mov byte ptr [rcx], al
    inc rcx
    jmp .Lje_next
.Lje_bs:                                // backslash + the char itself
    mov byte ptr [rcx], '\\'
    mov byte ptr [rcx + 1], al
    add rcx, 2
    jmp .Lje_next
.Lje_ctl:
    cmp eax, 10
    jne 1f
    mov eax, 'n'
    jmp .Lje_bs
1:  cmp eax, 13
    jne 2f
    mov eax, 'r'
    jmp .Lje_bs
2:  cmp eax, 9
    jne 3f
    mov eax, 't'
    jmp .Lje_bs
3:  mov dword ptr [rcx], 0x3030755C     // "\u00" (little-endian bytes \ u 0 0)
    mov r11d, eax
    shr r11d, 4
    mov r11b, byte ptr [r10 + r11]
    mov byte ptr [rcx + 4], r11b
    and eax, 15
    mov al, byte ptr [r10 + rax]
    mov byte ptr [rcx + 5], al
    add rcx, 6
.Lje_next:
    cmp rdx, r8
    jb .Lje_loop
.Lje_end:
    mov rax, rcx
    sub rax, r9
    ret

// ===========================================================================
// i64 utf16_to_utf8(u8* dst, i64 cap, const wchar_t* src)
// ===========================================================================
    .globl utf16_to_utf8
utf16_to_utf8:
    push rbx
    sub rsp, 64                         // 8 + 8 + 64 = 80
    mov rbx, rcx
    test rdx, rdx
    jle .Lu8_fail
    // WideCharToMultiByte(CP_UTF8, 0, src, -1, dst, cap, NULL, NULL)
    mov qword ptr [rsp + 32], rcx
    mov qword ptr [rsp + 40], rdx
    mov qword ptr [rsp + 48], 0
    mov qword ptr [rsp + 56], 0
    mov ecx, CP_UTF8
    mov rdx, 0
    mov r9, -1
    call qword ptr [rip + __imp_WideCharToMultiByte]
    test eax, eax
    jz .Lu8_fail
    dec eax                             // count included the NUL
    jmp .Lu8_ret
.Lu8_fail:
    xor eax, eax
    test rbx, rbx
    jz .Lu8_ret
    mov byte ptr [rbx], 0
.Lu8_ret:
    add rsp, 64
    pop rbx
    ret

// ===========================================================================
// message builder
// ===========================================================================
// void msg_n(const u8* p, i64 n): append (truncating at MSG_SIZE)
    .globl msg_n
msg_n:
    push rsi
    push rdi
    mov rsi, rcx
    mov rax, qword ptr [rip + msg_len]
    mov rcx, MSG_SIZE
    sub rcx, rax                        // room left
    cmp rdx, rcx
    jbe 1f
    mov rdx, rcx
1:  test rdx, rdx
    jle 2f
    lea rdi, [rip + msg_buf]
    add rdi, rax
    mov rcx, rdx
    add qword ptr [rip + msg_len], rdx
    rep movsb
2:  pop rdi
    pop rsi
    ret

    .globl msg_z
msg_z:
    sub rsp, 40
    call z_len
    mov rdx, rax
    call msg_n
    add rsp, 40
    ret

    .globl msg_nl
msg_nl:
    lea rcx, [rip + crlf_str]
    mov edx, 2
    jmp msg_n                           // tail call: msg_n returns to our caller

// void msg_w(const wchar_t* w): append as UTF-8
    .globl msg_w
msg_w:
    sub rsp, 40
    mov r8, rcx                         // src
    lea rcx, [rip + msg_buf]
    mov rax, qword ptr [rip + msg_len]
    add rcx, rax
    mov rdx, MSG_SIZE
    sub rdx, rax                        // cap = room left
    call utf16_to_utf8
    add qword ptr [rip + msg_len], rax
    add rsp, 40
    ret

    .globl msg_i64
msg_i64:
    sub rsp, 40
    mov rdx, rcx
    lea rcx, [rip + msg_tmp]
    call fmt_i64
    lea rcx, [rip + msg_tmp]
    mov rdx, rax
    call msg_n
    add rsp, 40
    ret

    .globl msg_u64
msg_u64:
    sub rsp, 40
    mov rdx, rcx
    lea rcx, [rip + msg_tmp]
    call fmt_u64
    lea rcx, [rip + msg_tmp]
    mov rdx, rax
    call msg_n
    add rsp, 40
    ret

    .globl msg_hex
msg_hex:
    sub rsp, 40
    mov rdx, rcx
    lea rcx, [rip + msg_tmp]
    call fmt_hex
    lea rcx, [rip + msg_tmp]
    mov rdx, rax
    call msg_n
    add rsp, 40
    ret

// void msg_f64(double v /* xmm0 */, i64 decimals /* rdx */)
    .globl msg_f64
msg_f64:
    sub rsp, 40
    movapd xmm1, xmm0
    mov r8, rdx
    lea rcx, [rip + msg_tmp]
    call fmt_f64
    lea rcx, [rip + msg_tmp]
    mov rdx, rax
    call msg_n
    add rsp, 40
    ret

    .globl msg_to_err
msg_to_err:
    sub rsp, 40
    lea rcx, [rip + msg_buf]
    mov rdx, qword ptr [rip + msg_len]
    call io_err
    mov qword ptr [rip + msg_len], 0
    add rsp, 40
    ret

    .globl msg_to_out
msg_to_out:
    sub rsp, 40
    lea rcx, [rip + msg_buf]
    mov rdx, qword ptr [rip + msg_len]
    call io_out
    mov qword ptr [rip + msg_len], 0
    add rsp, 40
    ret
