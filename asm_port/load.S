// load.S - LLMA / LLMQ checkpoint loader + KV-cache allocation (owner: loader agent)
//
// Exported (Microsoft x64 ABI):
//   i64  model_load(const wchar_t* path, Model* m, i64 flags)  -> 0 ok, 1 error
//            flags bit 0 (1): quantize an fp32 'LLMA' checkpoint to Q8 at load
//                             (via the kernels' mat_q8_from_f32)
//            flags bit 1 (2): keep the raw int8 copy of EVERY Q8 matrix (tests);
//                             by default the raw bytes of non-embedding matrices
//                             are given back to the OS once mat_q8_pack built the
//                             kernel layout (they are never read again)
//   i64  model_peek_magic(const wchar_t* path) -> first 4 bytes as u32,
//            0 if the file is shorter, -1 if it cannot be opened
//   void model_free(Model* m)        releases what model_load allocated itself
//   i64  kv_create(const Model*, KV*)              -> 0 ok, 1 out of memory
//   i64  kv_create_len(const Model*, KV*, i64 max_len)
//   void kv_free(KV*)
//   void q8_quant_percol(const float* w, i64 rows, i64 cols, i8* q, float* scale)
//            = quant.c quantize_int8_percol, bit for bit (used for the embedding
//              under flag 1: the embedding needs raw int8 + scale, not a packed layout)
//
// File format (port of ../llama.c read_header / read_windows / llama_config_valid /
// llama_model_load / q_llama_model_load):
//
//   u32 magic        'LLMA' = 0x414D4C4C (f32 weights) | 'LLMQ' = 0x514D4C4C (int8)
//   u32 version      2, 3 or 4
//   i32 cfg[8]       vocab, d_model, n_layers, n_heads, n_kv, head_dim, d_ff, max_seq
//   f32 rope_base, eps
//   u32 flags        ONLY in v4: bit0 QK-norm, bit1 windows present, bit2 tied head
//   i32 window[n_layers]   in v3, and in v4 when flags bit1
//   tensors in llama_collect's canonical order:
//     wte (vocab, d_model)
//     per layer: attn_norm, Wq, Wk, Wv, b_q, b_k, b_v, [q_norm, k_norm], Wo,
//                ffn_norm, Wgate, Wup, Wdown
//     final_norm, lm_head (d_model, vocab)   (lm_head absent when tied)
//   LLMA: every tensor is f32. LLMQ: 1-D tensors are f32; a 2-D (rows, cols)
//   tensor is f32 scale[cols] followed by int8 q[rows*cols] (row-major).
//
// How the load is made fast
// -------------------------
// 1. A dry "walk" over the canonical tensor list computes every tensor's file
//    offset and its slot in one of two arenas (vectors / matrices), each slot
//    64-byte aligned, and the total file size we expect.
// 2. Both arenas are allocated once, and the walk runs again to assign the Model's
//    pointers and record a READ PLAN: (file offset, destination, bytes) pieces of at
//    most 4 MB.
// 3. Several threads, each with its own file handle, pull pieces from the plan with
//    an atomic counter and ReadFile them (positional reads via OVERLAPPED offsets)
//    straight into their final place: no intermediate buffer, no second copy, and
//    the page faults of the fresh memory are spread over all cores.

    .intel_syntax noprefix
#include "model.inc"
#ifndef KV_SPARE
#define KV_SPARE    56                  // KV.spare (DESIGN.md: offset 56; not in model.inc yet)
#endif

#define LLMA_MAGIC        0x414D4C4C
#define LLMQ_MAGIC        0x514D4C4C
#define GPTC_MAGIC        0x43545047
#define FLAG_QK_NORM      1
#define FLAG_WINDOWS      2
#define FLAG_TIED         4
#define KNOWN_FLAGS       7
#define INT_MAX_          0x7FFFFFFF

#define GENERIC_READ      0x80000000
#define FILE_SHARE_READ   1
#define OPEN_EXISTING     3
#define FILE_ATTRIBUTE_NORMAL 0x80
#define FILE_FLAG_SEQUENTIAL_SCAN 0x08000000
#define INFINITE          0xFFFFFFFF

#define CHUNK             (4 << 20)     // read-plan piece size
#define MAX_THREADS       16
#define PLAN_ENTRY        24            // u64 file offset, u64 dst, u64 bytes

// tensor kinds in the walk tables
#define T_END   0
#define T_VEC   1                       // f32 vector (count = cols)
#define T_MAT   2                       // matrix (rows, cols) -> a Mat
#define T_QVEC  3                       // vector present only with QK-norm
// dimension codes -> index into ld_dims
#define C_ONE   0
#define C_D     1
#define C_QD    2
#define C_KVD   3
#define C_HD    4
#define C_DFF   5
#define C_V     6

    .section .rdata, "dr"
// One 4-byte entry per tensor of a layer, in file order:
//   kind, rows code, cols code, Layer field offset / 8
layer_tab:
    .byte T_VEC,  C_ONE, C_D,   L_ATTN_NORM / 8
    .byte T_MAT,  C_D,   C_QD,  L_WQ / 8
    .byte T_MAT,  C_D,   C_KVD, L_WK / 8
    .byte T_MAT,  C_D,   C_KVD, L_WV / 8
    .byte T_VEC,  C_ONE, C_QD,  L_BQ / 8
    .byte T_VEC,  C_ONE, C_KVD, L_BK / 8
    .byte T_VEC,  C_ONE, C_KVD, L_BV / 8
    .byte T_QVEC, C_ONE, C_HD,  L_QNORM / 8
    .byte T_QVEC, C_ONE, C_HD,  L_KNORM / 8
    .byte T_MAT,  C_QD,  C_D,   L_WO / 8
    .byte T_VEC,  C_ONE, C_D,   L_FFN_NORM / 8
    .byte T_MAT,  C_D,   C_DFF, L_WGATE / 8
    .byte T_MAT,  C_D,   C_DFF, L_WUP / 8
    .byte T_MAT,  C_DFF, C_D,   L_WDOWN / 8
    .byte T_END,  0,     0,     0

    .balign 32
abs_mask8:  .long 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF
            .long 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF
f_127:      .float 127.0
f_one:      .float 1.0
i_127:      .long 127
i_m127:     .long -127

s_err:        .asciz "Error: '"
s_cannot_open:.asciz "Error: cannot open '"
s_for_reading:.asciz "' for reading"
s_trunc_hdr:  .asciz "' is truncated (bad header)"
s_version:    .asciz "' is checkpoint version "
s_reads:      .asciz " (this build reads v2-v4)"
s_flags:      .asciz "' uses unsupported feature flags 0x"
s_not_llm:    .asciz "' is not an LLMA/LLMQ checkpoint"
s_tied_q:     .asciz "' is a tied-head LLMQ checkpoint; the int8 in-memory path needs an untied one (convert_llama.py --int8 writes that)"
s_bad_cfg:    .asciz "' has an invalid/unsupported config"
s_bad_win:    .asciz "' has a truncated/invalid sliding-window block"
s_trunc_w:    .asciz "' is truncated (bad weights)"
s_oom:        .asciz "Error: out of memory loading '"
s_quote:      .asciz "'"

    .bss
    .balign 8
ld_path:     .quad 0                    // wchar_t* path of the checkpoint
ld_model:    .quad 0                    // Model* being filled
ld_flags:    .quad 0
ld_q8file:   .quad 0                    // 1 = LLMQ (2-D tensors are int8)
ld_qk:       .quad 0
ld_tied:     .quad 0
ld_dims:     .space 8 * 8               // C_* -> dimension value
ld_pass:     .quad 0                    // 0 = sizing walk, 1 = assigning walk
ld_foff:     .quad 0                    // running file offset
ld_wstart:   .quad 0                    // file offset of the first tensor
ld_vcur:     .quad 0                    // vector arena cursor
ld_mcur:     .quad 0                    // matrix arena cursor
ld_vbase:    .quad 0
ld_mbase:    .quad 0
ld_plan:     .quad 0                    // PLAN_ENTRY * ld_npieces
ld_npieces:  .quad 0
ld_next:     .quad 0                    // next plan piece to read (atomic)
ld_err:      .quad 0                    // set by any reader thread on failure
ld_fsize:    .quad 0
ld_win:      .quad 0                    // temp i32 windows
    .balign 16
ld_hdr:      .space 64                  // raw header bytes

    .text

// ===========================================================================
// i64 ld_read_at(HANDLE h, u64 off, void* dst, i64 bytes) -> 0 ok, 1 short/failed
// Positional read: passing an OVERLAPPED with Offset/OffsetHigh to ReadFile on a
// normal (synchronous) handle reads from that offset and waits for completion.
// ===========================================================================
ld_read_at:
    push rbx
    push rsi
    push rdi
    push r12
    sub rsp, 88                         // 8 + 32 + 88 = 128
    // [rsp+32] 5th arg, [rsp+40] DWORD got, [rsp+48..80) OVERLAPPED
    mov rbx, rcx
    mov rsi, rdx
    mov rdi, r8
    mov r12, r9
.Lra_loop:
    test r12, r12
    jle .Lra_ok
    mov r8, r12
    cmp r8, 0x40000000
    jbe 1f
    mov r8d, 0x40000000
1:  xor eax, eax
    mov qword ptr [rsp + 48], rax       // Internal
    mov qword ptr [rsp + 56], rax       // InternalHigh
    mov qword ptr [rsp + 64], rsi       // Offset (low dword) + OffsetHigh (high dword)
    mov qword ptr [rsp + 72], rax       // hEvent
    mov dword ptr [rsp + 40], eax
    lea rax, [rsp + 48]
    mov qword ptr [rsp + 32], rax       // lpOverlapped
    mov rcx, rbx                        // ReadFile(h, dst, n, &got, &ov)
    mov rdx, rdi
    lea r9, [rsp + 40]
    call qword ptr [rip + __imp_ReadFile]
    test eax, eax
    jz .Lra_fail
    mov eax, dword ptr [rsp + 40]
    test eax, eax
    jz .Lra_fail
    add rsi, rax
    add rdi, rax
    sub r12, rax
    jmp .Lra_loop
.Lra_ok:
    xor eax, eax
    jmp .Lra_ret
.Lra_fail:
    mov eax, 1
.Lra_ret:
    add rsp, 88
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

// HANDLE ld_open(const wchar_t* path) -> handle or -1
ld_open:
    sub rsp, 72                         // 8 + 72 = 80; 7 args need [rsp+32..56)
    mov edx, GENERIC_READ
    mov r8d, FILE_SHARE_READ
    xor r9d, r9d
    mov qword ptr [rsp + 32], OPEN_EXISTING
    mov qword ptr [rsp + 40], FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN
    mov qword ptr [rsp + 48], 0
    call qword ptr [rip + __imp_CreateFileW]
    add rsp, 72
    ret

// ===========================================================================
// i64 model_peek_magic(const wchar_t* path)
// ===========================================================================
    .globl model_peek_magic
model_peek_magic:
    push rbx
    sub rsp, 48                         // 8 + 8 + 48 = 64; [rsp+40] = 4-byte buffer
    call ld_open
    cmp rax, -1
    je .Lpm_ret
    mov rbx, rax
    mov dword ptr [rsp + 40], 0
    mov rcx, rbx
    xor edx, edx
    lea r8, [rsp + 40]
    mov r9d, 4
    call ld_read_at
    mov ecx, dword ptr [rsp + 40]
    test eax, eax
    jz 1f
    xor ecx, ecx                        // shorter than 4 bytes -> 0
1:  mov dword ptr [rsp + 44], ecx
    mov rcx, rbx
    call qword ptr [rip + __imp_CloseHandle]
    mov eax, dword ptr [rsp + 44]       // zero-extended u32
.Lpm_ret:
    add rsp, 48
    pop rbx
    ret

// ===========================================================================
// read-plan building
// ===========================================================================
// ld_piece(u64 off, void* dst, i64 bytes): pass 0 counts the pieces, pass 1
// appends them to ld_plan. Leaf function (no calls), clobbers rax r9 r10 r11.
ld_piece:
    test r8, r8
    jle .Lpc_ret
    cmp qword ptr [rip + ld_pass], 0
    jne .Lpc_add
    lea rax, [r8 + CHUNK - 1]           // pieces += ceil(bytes / CHUNK)
    shr rax, 22                         // CHUNK = 2^22
    add qword ptr [rip + ld_npieces], rax
.Lpc_ret:
    ret
.Lpc_add:
    mov r10, qword ptr [rip + ld_plan]
    mov r11, qword ptr [rip + ld_npieces]
1:  mov r9, r8
    cmp r9, CHUNK
    jbe 2f
    mov r9d, CHUNK
2:  imul rax, r11, PLAN_ENTRY
    mov qword ptr [r10 + rax], rcx      // file offset
    mov qword ptr [r10 + rax + 8], rdx  // destination
    mov qword ptr [r10 + rax + 16], r9  // bytes
    inc r11
    add rcx, r9
    add rdx, r9
    sub r8, r9
    jnz 1b
    mov qword ptr [rip + ld_npieces], r11
    ret

// ld_tensor(kind ecx, rows rdx, cols r8, dest r9)
//   T_VEC: dest = address of a float* field;  T_MAT: dest = Mat*
ld_tensor:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    sub rsp, 32                         // 8 + 40 + 32 = 80
    mov rsi, rdx                        // rsi = rows
    mov rdi, r8                         // rdi = cols
    mov r12, r9                         // r12 = dest
    cmp ecx, T_VEC
    jne .Ltn_mat
    // ---- f32 vector of `cols` floats ----
    lea r13, [rdi*4]                    // bytes
    mov rbx, qword ptr [rip + ld_vbase]
    add rbx, qword ptr [rip + ld_vcur]  // slot address (meaningful in pass 1)
    cmp qword ptr [rip + ld_pass], 0
    je 1f
    mov qword ptr [r12], rbx
1:  mov rcx, qword ptr [rip + ld_foff]
    mov rdx, rbx
    mov r8, r13
    call ld_piece
    add qword ptr [rip + ld_foff], r13
    lea rax, [r13 + 63]
    and rax, -64
    add qword ptr [rip + ld_vcur], rax
    jmp .Ltn_ret
.Ltn_mat:
    cmp qword ptr [rip + ld_pass], 0
    je 2f
    mov qword ptr [r12 + MAT_K], rsi
    mov qword ptr [r12 + MAT_N], rdi
2:  cmp qword ptr [rip + ld_q8file], 0
    jne .Ltn_q8
    // ---- f32 matrix ----
    mov r13, rsi
    imul r13, rdi
    shl r13, 2                          // bytes = rows * cols * 4
    mov rbx, qword ptr [rip + ld_mbase]
    add rbx, qword ptr [rip + ld_mcur]
    cmp qword ptr [rip + ld_pass], 0
    je 3f
    mov qword ptr [r12 + MAT_KIND], MATK_F32
    mov qword ptr [r12 + MAT_DATA], rbx
3:  mov rcx, qword ptr [rip + ld_foff]
    mov rdx, rbx
    mov r8, r13
    call ld_piece
    add qword ptr [rip + ld_foff], r13
    lea rax, [r13 + 63]
    and rax, -64
    add qword ptr [rip + ld_mcur], rax
    jmp .Ltn_ret
.Ltn_q8:
    // ---- int8 matrix: scale[cols] f32, then q[rows*cols] ----
    lea r13, [rdi*4]                    // scale bytes
    mov rbx, qword ptr [rip + ld_mbase]
    add rbx, qword ptr [rip + ld_mcur]
    cmp qword ptr [rip + ld_pass], 0
    je 4f
    mov qword ptr [r12 + MAT_KIND], MATK_Q8
    mov qword ptr [r12 + MAT_SCALE], rbx
4:  mov rcx, qword ptr [rip + ld_foff]
    mov rdx, rbx
    mov r8, r13
    call ld_piece
    add qword ptr [rip + ld_foff], r13
    lea rax, [r13 + 63]
    and rax, -64
    add qword ptr [rip + ld_mcur], rax
    mov r13, rsi
    imul r13, rdi                       // int8 bytes
    mov rbx, qword ptr [rip + ld_mbase]
    add rbx, qword ptr [rip + ld_mcur]
    cmp qword ptr [rip + ld_pass], 0
    je 5f
    mov qword ptr [r12 + MAT_DATA], rbx
5:  mov rcx, qword ptr [rip + ld_foff]
    mov rdx, rbx
    mov r8, r13
    call ld_piece
    add qword ptr [rip + ld_foff], r13
    lea rax, [r13 + 63]
    and rax, -64
    add qword ptr [rip + ld_mcur], rax
.Ltn_ret:
    add rsp, 32
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

// ld_walk(): visit every tensor in canonical (file) order.
ld_walk:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    sub rsp, 32
    mov qword ptr [rip + ld_vcur], 0
    mov qword ptr [rip + ld_mcur], 0
    mov rax, qword ptr [rip + ld_wstart]
    mov qword ptr [rip + ld_foff], rax
    lea r13, [rip + ld_dims]
    mov rbx, qword ptr [rip + ld_model]
    // wte (vocab, d_model)
    mov ecx, T_MAT
    mov rdx, qword ptr [r13 + C_V*8]
    mov r8, qword ptr [r13 + C_D*8]
    lea r9, [rbx + M_WTE]
    call ld_tensor
    // layers
    xor r12d, r12d                      // r12 = layer index
.Lwk_layer:
    cmp r12, qword ptr [rbx + M_NLAYERS]
    jae .Lwk_tail
    mov rsi, r12
    shl rsi, 9                          // * L_SIZE (512)
    add rsi, qword ptr [rbx + M_LAYERS] // rsi = Layer*
    lea rdi, [rip + layer_tab]          // rdi = table cursor
.Lwk_ent:
    movzx ecx, byte ptr [rdi]
    cmp ecx, T_END
    je .Lwk_next
    cmp ecx, T_QVEC
    jne 1f
    cmp qword ptr [rip + ld_qk], 0
    je .Lwk_skip                        // no QK-norm: q_norm/k_norm not in the file
    mov ecx, T_VEC
1:  movzx eax, byte ptr [rdi + 1]
    mov rdx, qword ptr [r13 + rax*8]    // rows
    movzx eax, byte ptr [rdi + 2]
    mov r8, qword ptr [r13 + rax*8]     // cols
    movzx eax, byte ptr [rdi + 3]
    lea r9, [rsi + rax*8]               // field address
    call ld_tensor
.Lwk_skip:
    add rdi, 4
    jmp .Lwk_ent
.Lwk_next:
    inc r12
    jmp .Lwk_layer
.Lwk_tail:
    mov ecx, T_VEC                      // final_norm (d_model)
    mov edx, 1
    mov r8, qword ptr [r13 + C_D*8]
    lea r9, [rbx + M_FINALNORM]
    call ld_tensor
    cmp qword ptr [rip + ld_tied], 0
    jne 2f
    mov ecx, T_MAT                      // lm_head (d_model, vocab)
    mov rdx, qword ptr [r13 + C_D*8]
    mov r8, qword ptr [r13 + C_V*8]
    lea r9, [rbx + M_LMHEAD]
    call ld_tensor
2:  add rsp, 32
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

// ===========================================================================
// parallel execution of the read plan
// ===========================================================================
// DWORD WINAPI ld_worker(void*): one reader thread (the main thread runs one too)
ld_worker:
    push rbx
    push rsi
    sub rsp, 40                         // 8 + 16 + 40 = 64
    mov rcx, qword ptr [rip + ld_path]
    call ld_open
    cmp rax, -1
    je .Lwr_fail_noh
    mov rbx, rax
.Lwr_loop:
    cmp qword ptr [rip + ld_err], 0
    jne .Lwr_done
    mov eax, 1
    lock xadd qword ptr [rip + ld_next], rax    // rax = old value = my piece
    cmp rax, qword ptr [rip + ld_npieces]
    jae .Lwr_done
    imul rax, rax, PLAN_ENTRY
    mov rsi, qword ptr [rip + ld_plan]
    add rsi, rax
    mov rcx, rbx
    mov rdx, qword ptr [rsi]
    mov r8, qword ptr [rsi + 8]
    mov r9, qword ptr [rsi + 16]
    call ld_read_at
    test eax, eax
    jz .Lwr_loop
    lock or qword ptr [rip + ld_err], 1
.Lwr_done:
    mov rcx, rbx
    call qword ptr [rip + __imp_CloseHandle]
    jmp .Lwr_ret
.Lwr_fail_noh:
    lock or qword ptr [rip + ld_err], 1
.Lwr_ret:
    xor eax, eax
    add rsp, 40
    pop rsi
    pop rbx
    ret

// i64 ld_exec(void) -> 0 ok, nonzero = a read failed
ld_exec:
    push rbx
    push rsi
    push rdi
    sub rsp, 48 + 8*MAX_THREADS         // 8 + 24 + 48 + 128 = 208
    // handles live at [rsp + 48 ...]
    mov qword ptr [rip + ld_next], 0
    mov qword ptr [rip + ld_err], 0
    mov ecx, 0xFFFF                     // ALL_PROCESSOR_GROUPS
    call qword ptr [rip + __imp_GetActiveProcessorCount]
    mov esi, eax                        // rsi = threads wanted
    cmp rsi, MAX_THREADS
    jbe 1f
    mov esi, MAX_THREADS
1:  test rsi, rsi
    jnz 2f
    mov esi, 1
2:  mov rax, qword ptr [rip + ld_foff]
    sub rax, qword ptr [rip + ld_wstart]
    cmp rax, (64 << 20)                 // small model: not worth the threads
    jae 3f
    mov esi, 1
3:  cmp rsi, qword ptr [rip + ld_npieces]
    jbe 4f
    mov rsi, qword ptr [rip + ld_npieces]
4:  xor ebx, ebx                        // rbx = helper threads started
    dec rsi                             // the main thread is one of them
.Lex_spawn:
    cmp rbx, rsi
    jge .Lex_main
    xor ecx, ecx                        // CreateThread(NULL, 0, ld_worker, NULL, 0, NULL)
    mov edx, 65536                      // 64 KB stack is plenty
    lea r8, [rip + ld_worker]
    xor r9d, r9d
    mov qword ptr [rsp + 32], 0
    mov qword ptr [rsp + 40], 0
    call qword ptr [rip + __imp_CreateThread]
    test rax, rax
    jz .Lex_main                        // could not start more: work with what we have
    mov qword ptr [rsp + 48 + rbx*8], rax
    inc rbx
    jmp .Lex_spawn
.Lex_main:
    xor ecx, ecx
    call ld_worker
    test rbx, rbx
    jz .Lex_ret
    mov rcx, rbx                        // WaitForMultipleObjects(n, handles, TRUE, INFINITE)
    lea rdx, [rsp + 48]
    mov r8d, 1
    mov r9d, INFINITE
    call qword ptr [rip + __imp_WaitForMultipleObjects]
    xor edi, edi
5:  mov rcx, qword ptr [rsp + 48 + rdi*8]
    call qword ptr [rip + __imp_CloseHandle]
    inc rdi
    cmp rdi, rbx
    jb 5b
.Lex_ret:
    mov rax, qword ptr [rip + ld_err]
    add rsp, 48 + 8*MAX_THREADS
    pop rdi
    pop rsi
    pop rbx
    ret

// ===========================================================================
// error message helpers: "Error: '<path><tail>" (+ CRLF) to stderr
// ===========================================================================
// ld_msg_head(): appends "Error: '" + path
ld_msg_head:
    sub rsp, 40
    lea rcx, [rip + s_err]
    call msg_z
    mov rcx, qword ptr [rip + ld_path]
    call msg_w
    add rsp, 40
    ret

// ld_fail(const char* tail): full message, returns 1 in rax
ld_fail:
    push rbx
    sub rsp, 32
    mov rbx, rcx
    call ld_msg_head
    mov rcx, rbx
    call msg_z
    call msg_nl
    call msg_to_err
    mov eax, 1
    add rsp, 32
    pop rbx
    ret

// ===========================================================================
// i64 model_load(const wchar_t* path, Model* m, i64 flags)
// ===========================================================================
    .globl model_load
model_load:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    push r14
    push r15
    sub rsp, 48                         // 8 + 56 + 48 = 112; [rsp+40] = file size
    mov qword ptr [rip + ld_path], rcx
    mov qword ptr [rip + ld_model], rdx
    mov qword ptr [rip + ld_flags], r8
    mov rbx, rdx                        // rbx = Model* (kept for the whole function)
    mov qword ptr [rip + ld_vbase], 0
    mov qword ptr [rip + ld_mbase], 0
    mov qword ptr [rip + ld_plan], 0
    mov qword ptr [rip + ld_win], 0
    // zero the Model (256 bytes)
    mov rdi, rbx
    xor eax, eax
    mov ecx, M_SIZE
    rep stosb

    // ---- open, size, header ----
    mov rcx, qword ptr [rip + ld_path]
    call ld_open
    cmp rax, -1
    jne 1f
    lea rcx, [rip + s_cannot_open]      // "Error: cannot open '<path>' for reading"
    call msg_z
    mov rcx, qword ptr [rip + ld_path]
    call msg_w
    lea rcx, [rip + s_for_reading]
    call msg_z
    call msg_nl
    call msg_to_err
    jmp .Lml_fail
1:  mov r12, rax                        // r12 = handle (closed after the header)
    mov rcx, r12
    lea rdx, [rsp + 40]
    call qword ptr [rip + __imp_GetFileSizeEx]
    mov rax, qword ptr [rsp + 40]
    mov qword ptr [rip + ld_fsize], rax
    // header bytes available: min(file size, 52)
    mov r9, rax
    cmp r9, 52
    jbe 2f
    mov r9d, 52
2:  mov r13, r9                         // r13 = header bytes read
    lea rdi, [rip + ld_hdr]             // zero the header buffer first
    xor eax, eax
    mov ecx, 64
    rep stosb
    mov rcx, r12
    xor edx, edx
    lea r8, [rip + ld_hdr]
    call ld_read_at
    test eax, eax
    jz 3f
    xor r13d, r13d                      // read failed: treat as empty
3:  lea r14, [rip + ld_hdr]             // r14 = header bytes
    // truncation: need 48 bytes (52 for v4, whose flags word follows)
    cmp r13, 48
    jb .Lml_trunc_hdr
    mov eax, dword ptr [r14 + 4]        // version
    cmp eax, 4
    jne 4f
    cmp r13, 52
    jb .Lml_trunc_hdr
4:  // version must be 2, 3 or 4
    cmp eax, 2
    jb .Lml_bad_version
    cmp eax, 4
    ja .Lml_bad_version
    // flags (v4 only)
    xor r15d, r15d                      // r15 = flags
    cmp eax, 4
    jne 5f
    mov r15d, dword ptr [r14 + 48]
5:  test r15d, ~KNOWN_FLAGS
    jnz .Lml_bad_flags
    // magic
    mov eax, dword ptr [r14]
    xor ecx, ecx
    cmp eax, LLMA_MAGIC
    je 6f
    mov ecx, 1
    cmp eax, LLMQ_MAGIC
    jne .Lml_not_llm
6:  mov qword ptr [rip + ld_q8file], rcx
    // an int8 file with a tied head is rejected (as q_llama_model_load does)
    test ecx, ecx
    jz 7f
    test r15d, FLAG_TIED
    jnz .Lml_tied_q
7:  // ---- llama_config_valid ----
    // every cfg[i] in 1..1000000 (as signed int32)
    xor ecx, ecx
8:  movsxd rax, dword ptr [r14 + 8 + rcx*4]
    cmp rax, 1
    jl .Lml_bad_cfg
    cmp rax, 1000000
    jg .Lml_bad_cfg
    inc ecx
    cmp ecx, 8
    jb 8b
    // n_heads % n_kv == 0
    movsxd rax, dword ptr [r14 + 20]    // n_heads
    movsxd rcx, dword ptr [r14 + 24]    // n_kv
    cqo
    idiv rcx
    test rdx, rdx
    jnz .Lml_bad_cfg
    // qd = n_heads*head_dim, kvd = n_kv*head_dim, both <= 1e6
    movsxd rax, dword ptr [r14 + 20]
    movsxd rcx, dword ptr [r14 + 28]    // head_dim
    imul rax, rcx
    cmp rax, 1000000
    jg .Lml_bad_cfg
    mov rsi, rax                        // rsi = qd
    movsxd rax, dword ptr [r14 + 24]
    imul rax, rcx
    cmp rax, 1000000
    jg .Lml_bad_cfg
    mov rdi, rax                        // rdi = kvd
    // vocab*d_model, d_model*d_ff, qd*d_model all <= INT_MAX
    movsxd rax, dword ptr [r14 + 8]
    movsxd rcx, dword ptr [r14 + 12]
    imul rax, rcx
    cmp rax, INT_MAX_
    jg .Lml_bad_cfg
    movsxd rax, dword ptr [r14 + 32]    // d_ff
    imul rax, rcx
    cmp rax, INT_MAX_
    jg .Lml_bad_cfg
    mov rax, rsi
    imul rax, rcx
    cmp rax, INT_MAX_
    jg .Lml_bad_cfg

    // ---- fill the config into the Model ----
    xor ecx, ecx
9:  movsxd rax, dword ptr [r14 + 8 + rcx*4]
    mov qword ptr [rbx + rcx*8], rax    // M_VOCAB .. M_MAXSEQ are consecutive i64s
    inc ecx
    cmp ecx, 8
    jb 9b
    mov eax, dword ptr [r14 + 40]
    mov dword ptr [rbx + M_ROPEBASE], eax
    mov eax, dword ptr [r14 + 44]
    mov dword ptr [rbx + M_EPS], eax
    mov eax, r15d
    and eax, FLAG_QK_NORM
    mov qword ptr [rbx + M_QKNORM], rax
    mov qword ptr [rip + ld_qk], rax
    mov eax, r15d
    shr eax, 2
    and eax, 1
    mov qword ptr [rbx + M_TIED], rax
    mov qword ptr [rip + ld_tied], rax
    mov rax, qword ptr [rip + ld_q8file]
    mov qword ptr [rbx + M_QUANT], rax
    // dims for the walk
    lea rcx, [rip + ld_dims]
    mov qword ptr [rcx + C_ONE*8], 1
    mov rax, qword ptr [rbx + M_DMODEL]
    mov qword ptr [rcx + C_D*8], rax
    mov qword ptr [rcx + C_QD*8], rsi
    mov qword ptr [rcx + C_KVD*8], rdi
    mov rax, qword ptr [rbx + M_HEADDIM]
    mov qword ptr [rcx + C_HD*8], rax
    mov rax, qword ptr [rbx + M_DFF]
    mov qword ptr [rcx + C_DFF*8], rax
    mov rax, qword ptr [rbx + M_VOCAB]
    mov qword ptr [rcx + C_V*8], rax

    // ---- layers array ----
    mov rcx, qword ptr [rbx + M_NLAYERS]
    shl rcx, 9
    call mem_alloc
    test rax, rax
    jz .Lml_oom_h
    mov qword ptr [rbx + M_LAYERS], rax

    // ---- sliding windows (v3, or v4 with flag bit 1) ----
    mov r13d, 48                        // r13 = header size
    mov eax, dword ptr [r14 + 4]
    cmp eax, 4
    jne 10f
    mov r13d, 52
10: cmp eax, 3
    je 11f
    cmp eax, 4
    jne .Lml_win_done
    test r15d, FLAG_WINDOWS
    jz .Lml_win_done
11: mov rcx, qword ptr [rbx + M_NLAYERS]
    shl rcx, 2
    call mem_alloc
    test rax, rax
    jz .Lml_oom_h
    mov qword ptr [rip + ld_win], rax
    mov rcx, r12
    mov rdx, r13
    mov r8, rax
    mov r9, qword ptr [rbx + M_NLAYERS]
    shl r9, 2
    call ld_read_at
    test eax, eax
    jnz .Lml_bad_win
    // each window in 0..1000000, copied into Layer.window
    mov r8, qword ptr [rip + ld_win]
    mov r9, qword ptr [rbx + M_LAYERS]
    xor ecx, ecx
12: cmp rcx, qword ptr [rbx + M_NLAYERS]
    jae 13f
    movsxd rax, dword ptr [r8 + rcx*4]
    test rax, rax
    js .Lml_bad_win
    cmp rax, 1000000
    jg .Lml_bad_win
    mov rdx, rcx
    shl rdx, 9
    mov qword ptr [r9 + rdx + L_WINDOW], rax
    inc rcx
    jmp 12b
13: mov rax, qword ptr [rbx + M_NLAYERS]
    shl rax, 2
    add r13, rax
.Lml_win_done:
    mov qword ptr [rip + ld_wstart], r13
    mov rcx, r12                        // header done: the readers open their own handles
    call qword ptr [rip + __imp_CloseHandle]
    xor r12d, r12d

    // ---- pass 0: sizes ----
    mov qword ptr [rip + ld_pass], 0
    mov qword ptr [rip + ld_npieces], 0
    call ld_walk
    mov rax, qword ptr [rip + ld_foff]  // expected end of the file
    cmp rax, qword ptr [rip + ld_fsize]
    ja .Lml_trunc_w
    // ---- arenas + plan ----
    mov rcx, qword ptr [rip + ld_vcur]
    add rcx, 64
    call mem_alloc
    test rax, rax
    jz .Lml_oom
    mov qword ptr [rip + ld_vbase], rax
    mov rcx, qword ptr [rip + ld_mcur]
    add rcx, 64
    call mem_alloc
    test rax, rax
    jz .Lml_oom
    mov qword ptr [rip + ld_mbase], rax
    mov rcx, qword ptr [rip + ld_npieces]
    imul rcx, rcx, PLAN_ENTRY
    add rcx, 64
    call mem_alloc
    test rax, rax
    jz .Lml_oom
    mov qword ptr [rip + ld_plan], rax
    // ---- pass 1: assign pointers, record the plan ----
    mov qword ptr [rip + ld_pass], 1
    mov qword ptr [rip + ld_npieces], 0
    call ld_walk
    // ---- read everything ----
    call ld_exec
    test rax, rax
    jnz .Lml_trunc_w
    mov rcx, qword ptr [rip + ld_plan]
    call mem_free
    mov qword ptr [rip + ld_plan], 0
    mov rax, qword ptr [rip + ld_mbase]
    mov qword ptr [rbx + M_FILEBASE], rax
    mov rax, qword ptr [rip + ld_vbase]
    mov qword ptr [rbx + M_SPARE], rax

    // ---- post-processing per format ----
    cmp qword ptr [rip + ld_q8file], 0
    jne .Lml_post_q8
    test qword ptr [rip + ld_flags], 1
    jnz .Lml_post_quantize
    // fp32 kept as is; a tied head is wte read transposed (F32T)
    cmp qword ptr [rip + ld_tied], 0
    je .Lml_ok
    mov qword ptr [rbx + M_LMHEAD + MAT_KIND], MATK_F32T
    mov rax, qword ptr [rbx + M_DMODEL]
    mov qword ptr [rbx + M_LMHEAD + MAT_K], rax
    mov rax, qword ptr [rbx + M_VOCAB]
    mov qword ptr [rbx + M_LMHEAD + MAT_N], rax
    mov rax, qword ptr [rbx + M_WTE + MAT_DATA]
    mov qword ptr [rbx + M_LMHEAD + MAT_DATA], rax
    jmp .Lml_ok

.Lml_post_q8:
    // LLMQ: pack every Q8 matrix except the embedding (embedding = row lookup)
    xor r13d, r13d                      // r13 = layer
.Lml_pq_layer:
    cmp r13, qword ptr [rbx + M_NLAYERS]
    jae .Lml_pq_head
    mov rsi, r13
    shl rsi, 9
    add rsi, qword ptr [rbx + M_LAYERS]
    mov r14d, L_WQ                      // the 7 Mats are consecutive: L_WQ .. L_WDOWN
.Lml_pq_mat:
    lea rcx, [rsi + r14]
    call ld_pack
    add r14d, MAT_SIZE
    cmp r14d, L_WDOWN
    jbe .Lml_pq_mat
    inc r13
    jmp .Lml_pq_layer
.Lml_pq_head:
    lea rcx, [rbx + M_LMHEAD]
    call ld_pack
    jmp .Lml_ok

.Lml_post_quantize:
    // flag 1: fp32 LLMA -> Q8 in memory
    mov qword ptr [rbx + M_QUANT], 1
    // (a) tied head: quantize wte^T, built in a temporary (d_model, vocab) f32
    cmp qword ptr [rip + ld_tied], 0
    je .Lml_pz_wte
    mov rcx, qword ptr [rbx + M_VOCAB]
    imul rcx, qword ptr [rbx + M_DMODEL]
    shl rcx, 2
    call mem_alloc
    test rax, rax
    jz .Lml_oom
    mov r13, rax                        // r13 = transposed copy
    mov rcx, qword ptr [rbx + M_WTE + MAT_DATA]
    mov rdx, qword ptr [rbx + M_VOCAB]
    mov r8, qword ptr [rbx + M_DMODEL]
    mov r9, r13
    call ld_transpose
    lea rcx, [rbx + M_LMHEAD]
    mov rdx, r13
    mov r8, qword ptr [rbx + M_DMODEL]
    mov r9, qword ptr [rbx + M_VOCAB]
    call ld_from_f32
    mov rcx, r13
    call mem_free
.Lml_pz_wte:
    // (b) embedding: raw int8 + per-column scale, no packed layout
    mov rcx, qword ptr [rbx + M_VOCAB]
    imul rcx, qword ptr [rbx + M_DMODEL]
    call mem_alloc
    test rax, rax
    jz .Lml_oom
    mov r13, rax                        // r13 = int8 q
    mov rcx, qword ptr [rbx + M_DMODEL]
    shl rcx, 2
    call mem_alloc
    test rax, rax
    jz .Lml_oom
    mov r14, rax                        // r14 = scale
    mov rcx, qword ptr [rbx + M_WTE + MAT_DATA]
    mov rdx, qword ptr [rbx + M_VOCAB]
    mov r8, qword ptr [rbx + M_DMODEL]
    mov r9, r13
    mov qword ptr [rsp + 32], r14
    call q8_quant_percol
    mov qword ptr [rbx + M_WTE + MAT_KIND], MATK_Q8
    mov qword ptr [rbx + M_WTE + MAT_DATA], r13
    mov qword ptr [rbx + M_WTE + MAT_SCALE], r14
    // (c) every layer matrix, and an untied lm_head
    xor r13d, r13d
.Lml_pz_layer:
    cmp r13, qword ptr [rbx + M_NLAYERS]
    jae .Lml_pz_head
    mov rsi, r13
    shl rsi, 9
    add rsi, qword ptr [rbx + M_LAYERS]
    mov r14d, L_WQ
.Lml_pz_mat:
    lea rcx, [rsi + r14]
    mov rdx, qword ptr [rcx + MAT_DATA]
    mov r8, qword ptr [rcx + MAT_K]
    mov r9, qword ptr [rcx + MAT_N]
    call ld_from_f32
    add r14d, MAT_SIZE
    cmp r14d, L_WDOWN
    jbe .Lml_pz_mat
    inc r13
    jmp .Lml_pz_layer
.Lml_pz_head:
    cmp qword ptr [rip + ld_tied], 0
    jne 14f
    lea rcx, [rbx + M_LMHEAD]
    mov rdx, qword ptr [rcx + MAT_DATA]
    mov r8, qword ptr [rcx + MAT_K]
    mov r9, qword ptr [rcx + MAT_N]
    call ld_from_f32
14: // the f32 matrices are no longer needed
    mov rcx, qword ptr [rip + ld_mbase]
    call mem_free
    mov qword ptr [rip + ld_mbase], 0
    mov qword ptr [rbx + M_FILEBASE], 0
    jmp .Lml_ok

.Lml_ok:
    mov rcx, qword ptr [rip + ld_win]
    call mem_free
    xor eax, eax
    jmp .Lml_ret

    // ---- errors (r12 = still-open header handle, or 0) ----
.Lml_trunc_hdr:
    lea rcx, [rip + s_trunc_hdr]
    call ld_fail
    jmp .Lml_cleanup
.Lml_bad_version:
    call ld_msg_head
    lea rcx, [rip + s_version]
    call msg_z
    mov ecx, dword ptr [r14 + 4]
    call msg_u64
    lea rcx, [rip + s_reads]
    call msg_z
    call msg_nl
    call msg_to_err
    jmp .Lml_cleanup
.Lml_bad_flags:
    call ld_msg_head
    lea rcx, [rip + s_flags]
    call msg_z
    mov ecx, r15d
    call msg_hex
    call msg_nl
    call msg_to_err
    jmp .Lml_cleanup
.Lml_not_llm:
    lea rcx, [rip + s_not_llm]
    call ld_fail
    jmp .Lml_cleanup
.Lml_tied_q:
    lea rcx, [rip + s_tied_q]
    call ld_fail
    jmp .Lml_cleanup
.Lml_bad_cfg:
    lea rcx, [rip + s_bad_cfg]
    call ld_fail
    jmp .Lml_cleanup
.Lml_bad_win:
    lea rcx, [rip + s_bad_win]
    call ld_fail
    jmp .Lml_cleanup
.Lml_trunc_w:
    lea rcx, [rip + s_trunc_w]
    call ld_fail
    jmp .Lml_cleanup
.Lml_oom_h:
.Lml_oom:
    lea rcx, [rip + s_oom]
    call msg_z
    mov rcx, qword ptr [rip + ld_path]
    call msg_w
    lea rcx, [rip + s_quote]
    call msg_z
    call msg_nl
    call msg_to_err
.Lml_cleanup:
    test r12, r12
    jz 15f
    mov rcx, r12
    call qword ptr [rip + __imp_CloseHandle]
15: mov rcx, qword ptr [rip + ld_plan]
    call mem_free
    mov rcx, qword ptr [rip + ld_win]
    call mem_free
    mov rcx, qword ptr [rip + ld_vbase]
    call mem_free
    mov rcx, qword ptr [rip + ld_mbase]
    call mem_free
    mov rcx, qword ptr [rbx + M_LAYERS]
    call mem_free
    mov qword ptr [rbx + M_LAYERS], 0
.Lml_fail:
    mov eax, 1
.Lml_ret:
    add rsp, 48
    pop r15
    pop r14
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

// ld_pack(Mat*): mat_q8_pack, then (unless flags bit 1) give the raw int8 pages
// back to the OS -- but only when the kernel really built a separate packed copy.
ld_pack:
    push rbx
    sub rsp, 32
    mov rbx, rcx
    call mat_q8_pack
    test qword ptr [rip + ld_flags], 2
    jnz 1f
    mov rax, qword ptr [rbx + MAT_PACKED]
    test rax, rax
    jz 1f                               // no packed layout: data still needed
    mov rcx, qword ptr [rbx + MAT_DATA]
    mov rdx, qword ptr [rbx + MAT_K]
    imul rdx, qword ptr [rbx + MAT_N]   // rdx = int8 bytes
    cmp rax, rcx                        // packed inside [data, data + bytes)?
    jb 2f
    lea r8, [rcx + rdx]
    cmp rax, r8
    jb 1f                               // packing reused the raw buffer: keep it
2:  call mem_decommit
    mov qword ptr [rbx + MAT_DATA], 0
1:  add rsp, 32
    pop rbx
    ret

// ld_from_f32(Mat* m, const float* w, K, N): clear the Mat, then let the kernels
// quantize + pack it (mat_q8_from_f32 = quantize_int8_percol, then pack)
ld_from_f32:
    push rbx
    push rsi
    push rdi
    push r12
    sub rsp, 40                         // 8 + 32 + 40 = 80
    mov rbx, rcx
    mov rsi, rdx
    mov rdi, r8
    mov r12, r9
    xor eax, eax
    mov qword ptr [rbx + MAT_DATA], rax
    mov qword ptr [rbx + MAT_SCALE], rax
    mov qword ptr [rbx + MAT_PACKED], rax
    mov qword ptr [rbx + MAT_COLSUM], rax
    mov qword ptr [rbx + MAT_AUX], rax
    mov qword ptr [rbx + MAT_KIND], MATK_Q8
    mov qword ptr [rbx + MAT_K], rdi
    mov qword ptr [rbx + MAT_N], r12
    mov rcx, rbx
    mov rdx, rsi
    mov r8, rdi
    mov r9, r12
    call mat_q8_from_f32
    add rsp, 40
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

// ld_transpose(const float* src, rows, cols, float* dst): dst[j*rows+i] = src[i*cols+j]
// Done in blocks of 32 source rows so the writes stay within a few cache lines.
ld_transpose:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    push r14
    // rcx = src, rdx = rows, r8 = cols, r9 = dst
    xor r10d, r10d                      // r10 = i0 (block start row)
.Ltr_block:
    cmp r10, rdx
    jae .Ltr_done
    lea r11, [r10 + 32]                 // r11 = i1 = min(i0 + 32, rows)
    cmp r11, rdx
    jbe 1f
    mov r11, rdx
1:  xor ebx, ebx                        // rbx = j
.Ltr_col:
    cmp rbx, r8
    jae .Ltr_next
    mov rsi, r10                        // rsi = i
    mov r12, rbx
    imul r12, rdx                       // r12 = j*rows
2:  mov rax, rsi
    imul rax, r8
    add rax, rbx                        // i*cols + j
    mov r13d, dword ptr [rcx + rax*4]
    lea r14, [r12 + rsi]
    mov dword ptr [r9 + r14*4], r13d
    inc rsi
    cmp rsi, r11
    jb 2b
    inc rbx
    jmp .Ltr_col
.Ltr_next:
    mov r10, r11
    jmp .Ltr_block
.Ltr_done:
    pop r14
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

// ===========================================================================
// void q8_quant_percol(const float* w, i64 rows, i64 cols, i8* q, float* scale)
// Bit-exact port of quant.c quantize_int8_percol:
//   mx[j]    = max_i |w[i][j]|
//   scale[j] = mx > 0 ? mx / 127 : 1        (float division)
//   q[i][j]  = clamp(lrintf(w[i][j] * (1/scale[j])), -127, 127)
// The C loop walks a column at a time; here we walk ROWS (cache friendly) and do
// 8 columns per AVX2 instruction. Every element sees exactly the same float
// operations (max is order-independent; the multiply and the round-to-nearest-
// even conversion are per element), so the result is identical.
// Only ymm0-ymm5 are used, so no callee-saved xmm register needs saving.
// ===========================================================================
    .globl q8_quant_percol
q8_quant_percol:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    push r14
    push r15
    sub rsp, 48                         // 8 + 56 + 48 = 112
    mov rbx, rcx                        // rbx = w
    mov r12, rdx                        // r12 = rows
    mov r13, r8                         // r13 = cols
    mov r14, r9                         // r14 = q
    mov r15, qword ptr [rsp + 112 + 32] // 5th argument: scale
    lea rcx, [r13*4]
    call mem_alloc                      // inv[cols]
    mov rsi, rax                        // rsi = inv
    test rsi, rsi
    jz .Lqq_ret
    // scale[] = 0 (it holds the running column maxima first)
    xor ecx, ecx
1:  cmp rcx, r13
    jae 2f
    mov dword ptr [r15 + rcx*4], 0
    inc rcx
    jmp 1b
2:  vmovups ymm5, ymmword ptr [rip + abs_mask8]
    // ---- pass 1: column maxima of |w| ----
    xor r8d, r8d                        // r8 = row
    mov r9, rbx                         // r9 = row pointer
.Lqq_max_row:
    cmp r8, r12
    jae .Lqq_scales
    xor ecx, ecx                        // rcx = column
3:  lea rax, [rcx + 8]
    cmp rax, r13
    ja 4f
    vandps ymm0, ymm5, ymmword ptr [r9 + rcx*4]     // |w|
    vmaxps ymm0, ymm0, ymmword ptr [r15 + rcx*4]
    vmovups ymmword ptr [r15 + rcx*4], ymm0
    add rcx, 8
    jmp 3b
4:  cmp rcx, r13                        // scalar tail
    jae 5f
    vmovss xmm0, dword ptr [r9 + rcx*4]
    vandps xmm0, xmm0, xmm5
    vmaxss xmm0, xmm0, dword ptr [r15 + rcx*4]
    vmovss dword ptr [r15 + rcx*4], xmm0
    inc rcx
    jmp 4b
5:  inc r8
    lea r9, [r9 + r13*4]
    jmp .Lqq_max_row
    // ---- scales and their inverses (scalar, exactly as the C code) ----
.Lqq_scales:
    vmovss xmm3, dword ptr [rip + f_127]
    vmovss xmm4, dword ptr [rip + f_one]
    vxorps xmm2, xmm2, xmm2
    xor ecx, ecx
6:  cmp rcx, r13
    jae 8f
    vmovss xmm0, dword ptr [r15 + rcx*4]            // mx
    vcomiss xmm0, xmm2
    jbe 7f                                          // mx <= 0 (or NaN): s = 1
    vdivss xmm0, xmm0, xmm3                         // s = mx / 127
    jmp 9f
7:  vmovaps xmm0, xmm4
9:  vmovss dword ptr [r15 + rcx*4], xmm0
    vdivss xmm1, xmm4, xmm0                         // inv = 1 / s
    vmovss dword ptr [rsi + rcx*4], xmm1
    inc rcx
    jmp 6b
    // ---- pass 2: quantize ----
8:  vpbroadcastd ymm3, dword ptr [rip + i_127]
    vpbroadcastd ymm4, dword ptr [rip + i_m127]
    xor r8d, r8d                        // row
    mov r9, rbx                         // row pointer (f32)
    mov r10, r14                        // row pointer (int8)
.Lqq_q_row:
    cmp r8, r12
    jae .Lqq_done
    xor ecx, ecx
10: lea rax, [rcx + 8]
    cmp rax, r13
    ja 11f
    vmovups ymm0, ymmword ptr [r9 + rcx*4]
    vmulps ymm0, ymm0, ymmword ptr [rsi + rcx*4]    // w * inv
    vcvtps2dq ymm0, ymm0                            // lrintf: round to nearest even
    vpminsd ymm0, ymm0, ymm3                        // clamp to [-127, 127]
    vpmaxsd ymm0, ymm0, ymm4
    vextracti128 xmm1, ymm0, 1
    vpackssdw xmm0, xmm0, xmm1                      // 8 x i32 -> 8 x i16
    vpacksswb xmm0, xmm0, xmm0                      // -> 8 x i8 (low 8 bytes)
    vmovq qword ptr [r10 + rcx], xmm0
    add rcx, 8
    jmp 10b
11: cmp rcx, r13                                    // scalar tail
    jae 12f
    vmovss xmm0, dword ptr [r9 + rcx*4]
    vmulss xmm0, xmm0, dword ptr [rsi + rcx*4]
    vcvtss2si eax, xmm0
    cmp eax, 127
    jle 13f
    mov eax, 127
13: cmp eax, -127
    jge 14f
    mov eax, -127
14: mov byte ptr [r10 + rcx], al
    inc rcx
    jmp 11b
12: inc r8
    lea r9, [r9 + r13*4]
    add r10, r13
    jmp .Lqq_q_row
.Lqq_done:
    vzeroupper
    mov rcx, rsi
    call mem_free
.Lqq_ret:
    add rsp, 48
    pop r15
    pop r14
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

// ===========================================================================
// void model_free(Model* m): frees the arenas, the layers and (flag 1) the
// embedding copy. Packed kernel buffers belong to the kernels and are not freed.
// ===========================================================================
    .globl model_free
model_free:
    push rbx
    sub rsp, 32
    mov rbx, rcx
    mov rcx, qword ptr [rbx + M_FILEBASE]
    test rcx, rcx
    jnz 1f
    // no matrix arena (flag 1): the embedding was allocated separately
    cmp qword ptr [rbx + M_WTE + MAT_KIND], MATK_Q8
    jne 2f
    mov rcx, qword ptr [rbx + M_WTE + MAT_DATA]
    call mem_free
    mov rcx, qword ptr [rbx + M_WTE + MAT_SCALE]
    call mem_free
    jmp 2f
1:  call mem_free
2:  mov rcx, qword ptr [rbx + M_SPARE]
    call mem_free
    mov rcx, qword ptr [rbx + M_LAYERS]
    call mem_free
    xor eax, eax
    mov qword ptr [rbx + M_FILEBASE], rax
    mov qword ptr [rbx + M_SPARE], rax
    mov qword ptr [rbx + M_LAYERS], rax
    add rsp, 32
    pop rbx
    ret

// ===========================================================================
// KV cache (port of llama.c kv_create_ex / llama_kv_create_len)
//   max_len outside 1..max_seq means max_seq. Layer l gets
//   cap[l] = (window > 0 && window < max_len) ? window : max_len   ring slots,
//   k[l] and v[l] = cap[l] * kv_dim floats each (64-byte aligned slices of one
//   arena, which KV.spare remembers for kv_free). cap/k/v are one i64[3*n_layers]
//   block: cap at +0, k at +8*n_layers, v at +16*n_layers.
// ===========================================================================
    .globl kv_create
kv_create:
    mov r8, qword ptr [rcx + M_MAXSEQ]
    jmp kv_create_len                   // tail call

    .globl kv_create_len
kv_create_len:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    push r14
    sub rsp, 40                         // 8 + 48 + 40 = 96
    mov rbx, rcx                        // rbx = Model*
    mov rsi, rdx                        // rsi = KV*
    mov r12, r8                         // r12 = max_len
    mov rax, qword ptr [rbx + M_MAXSEQ]
    cmp r12, 1
    jl 1f
    cmp r12, rax
    jle 2f
1:  mov r12, rax
2:  mov rdi, rsi                        // zero the KV struct
    xor eax, eax
    mov ecx, KV_SIZE
    rep stosb
    mov r13, qword ptr [rbx + M_NLAYERS]    // r13 = n_layers
    mov r14, qword ptr [rbx + M_NKV]
    imul r14, qword ptr [rbx + M_HEADDIM]   // r14 = kv_dim
    mov qword ptr [rsi + KV_NLAYERS], r13
    mov qword ptr [rsi + KV_MAXSEQ], r12
    mov qword ptr [rsi + KV_KVDIM], r14
    lea rcx, [r13 + r13*2]
    shl rcx, 3                          // 24 bytes per layer
    call mem_alloc
    test rax, rax
    jz .Lkv_fail
    mov qword ptr [rsi + KV_CAP], rax
    lea rcx, [rax + r13*8]
    mov qword ptr [rsi + KV_K], rcx
    lea rcx, [rax + r13*8]
    lea rcx, [rcx + r13*8]
    mov qword ptr [rsi + KV_V], rcx
    // caps and the arena size
    mov r8, qword ptr [rbx + M_LAYERS]
    mov r9, rax                         // r9 = cap[]
    xor ecx, ecx
    xor edi, edi                        // rdi = arena bytes
3:  cmp rcx, r13
    jae 5f
    mov rdx, rcx
    shl rdx, 9
    mov rax, qword ptr [r8 + rdx + L_WINDOW]
    test rax, rax
    jle 4f
    cmp rax, r12
    jl 6f
4:  mov rax, r12
6:  mov qword ptr [r9 + rcx*8], rax
    imul rax, r14
    shl rax, 2
    add rax, 63
    and rax, -64
    lea rdi, [rdi + rax*2]              // k and v
    inc rcx
    jmp 3b
5:  lea rcx, [rdi + 64]
    call mem_alloc
    test rax, rax
    jz .Lkv_fail
    mov qword ptr [rsi + KV_SPARE], rax
    // slice the arena: k[l], v[l]
    mov r9, qword ptr [rsi + KV_CAP]
    mov r10, qword ptr [rsi + KV_K]
    mov r11, qword ptr [rsi + KV_V]
    xor ecx, ecx
7:  cmp rcx, r13
    jae 8f
    mov rdx, qword ptr [r9 + rcx*8]
    imul rdx, r14
    shl rdx, 2
    add rdx, 63
    and rdx, -64
    mov qword ptr [r10 + rcx*8], rax
    add rax, rdx
    mov qword ptr [r11 + rcx*8], rax
    add rax, rdx
    inc rcx
    jmp 7b
8:  xor eax, eax
    jmp .Lkv_ret
.Lkv_fail:
    mov rcx, qword ptr [rsi + KV_CAP]
    call mem_free
    mov qword ptr [rsi + KV_CAP], 0
    mov eax, 1
.Lkv_ret:
    add rsp, 40
    pop r14
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

    .globl kv_free
kv_free:
    push rbx
    sub rsp, 32
    mov rbx, rcx
    mov rcx, qword ptr [rbx + KV_SPARE]
    call mem_free
    mov rcx, qword ptr [rbx + KV_CAP]
    call mem_free
    xor eax, eax
    mov qword ptr [rbx + KV_SPARE], rax
    mov qword ptr [rbx + KV_CAP], rax
    mov qword ptr [rbx + KV_K], rax
    mov qword ptr [rbx + KV_V], rax
    add rsp, 32
    pop rbx
    ret
