// tok.S - byte-level BPE tokenizer in pure x86-64 assembly (Windows x64 ABI).
//
// A line-by-line port of ../hftok.c (hftok_load / hftok_encode / hftok_decode /
// hftok_is_eog). The output is identical to the C code, id for id; the internals
// are re-organised for speed:
//
//   * the whole .tok file is read into ONE buffer and every token / special /
//     system-prompt string just points into it (no per-token malloc);
//   * each vocab entry caches the FNV-1a hash of its bytes, so the hash of the
//     concatenation a||b is "continue FNV from hash(a) over the bytes of b" --
//     no copy into a scratch buffer as the C code does;
//   * the BPE loop keeps an array of pair ranks and after a merge recomputes only
//     the two pairs next to it (the C code re-looks-up every pair every round);
//   * Unicode classes (\p{L}, \p{N}, \s) come from a 0x110000-byte table built at
//     load time (one load per char) instead of a binary search per char;
//   * two caches inside the Tok: a WORD cache (pretoken bytes -> its ids; a hit
//     skips BPE entirely) and a PAIR cache ((a, b) -> merged id). Both are exact
//     (keys are compared in full). Consequence: tok_encode mutates the Tok, so do
//     not call it on the SAME Tok from two threads at once.
//
// Exported (all Microsoft x64 calling convention; ints are 64-bit):
//   Tok* tok_load(const wchar_t* path)                 -> 0 on any error
//   i64  tok_encode(Tok*, const u8* text, i64 len, i32* ids, i64 cap)
//   i64  tok_piece(Tok*, i64 id, u8* out)              -> byte count (no NUL written)
//   i64  tok_is_eog(Tok*, i64 id)                      -> 1 / 0
//   i64  tok_vocab(Tok*)                               -> n_vocab
//   i64  tok_max_piece(Tok*)                           -> longest token in bytes
//   void tok_free(Tok*)                                -> releases everything
//
// ---------------------------------------------------------------------------
// Quick reminder of the Windows x64 ABI used everywhere below:
//   args: rcx, rdx, r8, r9, then the stack ([rsp+40] at function entry, because
//   [rsp] is the return address and [rsp+8..40) is the caller's 32-byte "shadow
//   space"); return value in rax. A function may freely clobber rax rcx rdx r8-r11
//   ("volatile"); it must give back rbx rbp rdi rsi r12-r15 unchanged
//   ("non-volatile", we push/pop them). Before every `call` the stack pointer
//   must be a multiple of 16 and the callee owns 32 bytes above it (shadow space).
// ---------------------------------------------------------------------------

    .intel_syntax noprefix

// ===== Tok structure (allocated zeroed with HeapAlloc) =====================
#define TK_NVOCAB    0      // i64  number of tokens
#define TK_EOS       8      // i64  end-of-text id (sign-extended i32, -1 = none)
#define TK_BOS       16     // i64  begin-of-sequence id (-1 = none)
#define TK_PRETOK    24     // i64  pretokenizer kind: 0 GPT-2 (ASCII classes), 1 Qwen2, 2 SmolLM2
#define TK_TMPL      32     // i64  chat template kind (HFTOK_TMPL_*)
#define TK_NSPECIAL  40     // i64  number of special tokens
#define TK_TOKS      48     // TokEnt* [n_vocab] (see below)
#define TK_HMAP      56     // {i32 id, u32 fnv}* open-addressing hash: slot -> id (-1 = empty) + its hash
#define TK_HMASK     64     // i64   hash capacity - 1 (capacity is a power of two)
#define TK_SPECIALS  72     // SpEnt* [n_special]
#define TK_SYSPTR    80     // u8*  default system prompt (points into the file buffer, no NUL)
#define TK_SYSLEN    88     // i64  its length
#define TK_UCLASS    96     // u8*  [0x110000] class bits of every code point
#define TK_FILEBUF   104    // u8*  the whole .tok file (owner of all token bytes)
#define TK_MAXLEN    112    // i64  longest token (bytes)
#define TK_HEAP      120    // HANDLE process heap used for the allocations
#define TK_B2ID      128    // i32  byte2id[256]: id of each single-byte token, -1 = none
#define TK_WCACHE    1152   // WcEnt* word cache (see below)
#define TK_PCACHE    1160   // PcEnt* pair cache (see below)
#define TK_SIZE      1168

// Pair cache: merged id of a pair of ids (a, b), direct-mapped by a multiplicative
// hash of the 64-bit key (a << 32 | b). Entry = { u64 key | bit 63, i32 rank, pad };
// bit 63 is always set in a stored key, so an all-zero entry is "empty".
#define PC_BITS      16
#define PC_ENTRIES   (1 << PC_BITS)

// Word cache: the ids a pretoken encodes to depend only on its bytes, so the
// encoder remembers recent pretokens of 2..WC_KEYMAX bytes in a direct-mapped
// table indexed by the FNV-1a hash of their bytes. (This makes tok_encode NOT
// safe to call concurrently on the same Tok from several threads.)
// Entries are 256 bytes, so long whitespace runs (indentation: CR LF + 40
// spaces is ONE pretoken, and BPE on it is the most expensive case) are cached too.
#define WC_ENTRIES   32768  // 8 MB, reserved+committed by VirtualAlloc; pages are
#define WC_SHIFT     8      // only backed by RAM once touched. 256-byte entries.
#define WC_KEYMAX    120
#define WC_IDSMAX    32
#define WC_HASH      0      // u32  FNV-1a of the key
#define WC_KLEN      4      // u16  key length (0 = empty slot)
#define WC_NIDS      6      // u16  number of ids
#define WC_KEY       8      // u8   key[120]
#define WC_IDS       128    // i32  ids[32] (pretokens giving more ids are not cached)

// TokEnt: 16 bytes per vocab id
#define TE_PTR       0      // u8*  bytes of the token
#define TE_LEN       8      // u32  byte length
#define TE_FNV       12     // u32  FNV-1a hash of the bytes (unmasked)
// SpEnt: 16 bytes per special token
#define SP_PTR       0      // u8*  literal string, e.g. "<|im_start|>"
#define SP_ID        8      // i32  vocab id it encodes to
#define SP_LEN       12     // u16  literal length
#define SP_EOG       14     // u8   1 = ends generation

// class bits stored in the Unicode table
#define C_L          1      // letter  \p{L}
#define C_N          2      // number  \p{N}
#define C_S          4      // space   \s
#define UCLASS_SIZE  0x110000

#define NO_RANK      0x7FFFFFFF   // "this pair does not merge" (C: INT_MAX)
#define FNV_BASIS    2166136261
#define FNV_PRIME    16777619

// a few ASCII characters (numeric, to avoid quoting issues in the preprocessor)
#define CH_TAB   0x09
#define CH_LF    0x0A
#define CH_CR    0x0D
#define CH_SP    0x20
#define CH_QUOTE 0x27   // '
#define CH_LT    0x3C   // <
#define CH_d     0x64
#define CH_e     0x65
#define CH_l     0x6C
#define CH_m     0x6D
#define CH_r     0x72
#define CH_s     0x73
#define CH_t     0x74
#define CH_v     0x76

// Win32 constants
#define GENERIC_READ          0x80000000
#define FILE_SHARE_READ       1
#define OPEN_EXISTING         3
#define FILE_ATTRIBUTE_NORMAL 0x80
#define HEAP_ZERO_MEMORY      8
#define MEM_COMMIT_RESERVE    0x3000
#define MEM_RELEASE           0x8000
#define PAGE_READWRITE        4

    .text

// ============================================================================
// Tok* tok_load(const wchar_t* path)
// ============================================================================
// File format (all little-endian), exactly what hftok_load accepts:
//   "HFTK" u32 ver, u32 n (1..5000000), i32 eos
//   n times: u16 len, len bytes
//   ver >= 2: i32 bos, u8 template_kind, u16 syslen, syslen bytes,
//             u32 n_special (<= 100000), n_special times: i32 id, u8 eog, u16 len, bytes
//   ver >= 3: u8 pretokenizer kind (0..2, anything else is an error)
//
// Register use in the body:
//   rbx = Tok*        r13 = process heap      r14 = file handle (while open)
//   rsi = parse cursor rdi = end of file data  r12 = version   r15 = scratch
// NEED k: jump to .Lbad unless k more bytes are left in the file (cursor+k <= end).
.macro NEED k
    lea rax, [rsi + \k]
    cmp rax, rdi
    ja .Lld_bad
.endm

    .globl tok_load
    .p2align 4
tok_load:
    push rbp
    mov rbp, rsp
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    push r14
    push r15
    // 8 pushes so far (incl. rbp) + return address = 72 bytes: rsp is 8 mod 16.
    // 88 more makes it 16-aligned and leaves: [rsp+0..32) shadow space for the
    // callees, [rsp+32..56) stack args 5..7, [rsp+64] and [rsp+72] two locals.
    sub rsp, 88

    mov r12, rcx                       // path (until the file is open)
    xor ebx, ebx                       // no Tok yet
    mov r14, -1                        // no file handle yet

    call qword ptr [rip + __imp_GetProcessHeap]
    mov r13, rax

    // --- the Tok struct itself, zero-filled ---
    mov rcx, r13
    mov edx, HEAP_ZERO_MEMORY
    mov r8d, TK_SIZE
    call qword ptr [rip + __imp_HeapAlloc]
    test rax, rax
    jz .Lld_fail
    mov rbx, rax
    mov [rbx + TK_HEAP], r13

    // --- open the file: CreateFileW(path, GENERIC_READ, FILE_SHARE_READ, NULL,
    //                                 OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL) ---
    mov rcx, r12
    mov edx, GENERIC_READ
    mov r8d, FILE_SHARE_READ
    xor r9d, r9d
    mov qword ptr [rsp + 32], OPEN_EXISTING
    mov qword ptr [rsp + 40], FILE_ATTRIBUTE_NORMAL
    mov qword ptr [rsp + 48], 0
    call qword ptr [rip + __imp_CreateFileW]
    cmp rax, -1                        // INVALID_HANDLE_VALUE
    je .Lld_fail
    mov r14, rax

    // --- size ---
    mov rcx, r14
    lea rdx, [rsp + 64]
    call qword ptr [rip + __imp_GetFileSizeEx]
    test eax, eax
    jz .Lld_fail
    mov r15, [rsp + 64]                // r15 = file size in bytes

    // --- buffer for the whole file (+16 slack, never read) ---
    mov rcx, r13
    xor edx, edx
    lea r8, [r15 + 16]
    call qword ptr [rip + __imp_HeapAlloc]
    test rax, rax
    jz .Lld_fail
    mov [rbx + TK_FILEBUF], rax

    // --- read it: ReadFile moves at most 4 GB per call, so loop in 1 GB pieces ---
    xor edi, edi                       // rdi = bytes read so far
.Lld_read:
    cmp rdi, r15
    jae .Lld_read_done
    mov r8, r15
    sub r8, rdi                        // bytes still wanted
    mov eax, 0x40000000
    cmp r8, rax
    cmova r8, rax                      // min(wanted, 1 GB)
    mov rcx, r14
    mov rdx, [rbx + TK_FILEBUF]
    add rdx, rdi                       // destination
    lea r9, [rsp + 72]                 // &bytes_read (DWORD)
    mov qword ptr [rsp + 32], 0        // lpOverlapped = NULL
    call qword ptr [rip + __imp_ReadFile]
    test eax, eax
    jz .Lld_fail
    mov eax, dword ptr [rsp + 72]
    test eax, eax
    jz .Lld_fail                       // file shrank under us
    add rdi, rax
    jmp .Lld_read
.Lld_read_done:
    mov rcx, r14
    call qword ptr [rip + __imp_CloseHandle]
    mov r14, -1

    // ---------------- parse ----------------
    mov rsi, [rbx + TK_FILEBUF]        // cursor
    lea rdi, [rsi + r15]               // end

    NEED 16
    cmp dword ptr [rsi], 0x4B544648    // "HFTK" read as a little-endian u32
    jne .Lld_bad
    mov r12d, dword ptr [rsi + 4]      // ver (u32, zero-extended into r12)
    mov eax, dword ptr [rsi + 8]       // n
    test eax, eax
    jz .Lld_bad
    cmp eax, 5000000
    ja .Lld_bad
    mov [rbx + TK_NVOCAB], rax
    movsxd rax, dword ptr [rsi + 12]   // eos (signed)
    mov [rbx + TK_EOS], rax
    mov qword ptr [rbx + TK_BOS], -1
    add rsi, 16

    // byte2id[] = -1
    xor ecx, ecx
.Lld_b2i:
    mov dword ptr [rbx + TK_B2ID + rcx*4], -1
    inc ecx
    cmp ecx, 256
    jb .Lld_b2i

    // TokEnt array
    mov rcx, r13
    xor edx, edx
    mov r8, [rbx + TK_NVOCAB]
    shl r8, 4                          // 16 bytes per entry
    call qword ptr [rip + __imp_HeapAlloc]
    test rax, rax
    jz .Lld_fail
    mov [rbx + TK_TOKS], rax

    // ---- the n tokens: u16 len + bytes each ----
    mov r8, rax                        // r8  = &toks[i]
    xor r9d, r9d                       // r9  = i
.Lld_tok:
    cmp r9, [rbx + TK_NVOCAB]
    jae .Lld_tok_done
    NEED 2
    movzx ecx, word ptr [rsi]          // len
    add rsi, 2
    lea rax, [rsi + rcx]
    cmp rax, rdi
    ja .Lld_bad                        // truncated bytes
    mov [r8 + TE_PTR], rsi
    mov [r8 + TE_LEN], ecx
    cmp rcx, [rbx + TK_MAXLEN]
    jbe 1f
    mov [rbx + TK_MAXLEN], rcx
1:
    // FNV-1a over the bytes: h = basis; for each byte b: h ^= b; h *= prime
    mov eax, FNV_BASIS
    xor edx, edx                       // byte index
2:  cmp edx, ecx
    jae 3f
    movzx r10d, byte ptr [rsi + rdx]
    xor eax, r10d
    imul eax, eax, FNV_PRIME
    inc edx
    jmp 2b
3:  mov [r8 + TE_FNV], eax
    // a 1-byte token registers itself in byte2id (a later one overwrites, as in C)
    cmp ecx, 1
    jne 4f
    movzx eax, byte ptr [rsi]
    mov [rbx + TK_B2ID + rax*4], r9d
4:  add rsi, rcx
    add r8, 16
    inc r9
    jmp .Lld_tok
.Lld_tok_done:

    // ---- v2 tail ----
    cmp r12d, 2
    jb .Lld_v3
    NEED 7
    movsxd rax, dword ptr [rsi]        // bos
    mov [rbx + TK_BOS], rax
    movzx eax, byte ptr [rsi + 4]      // template kind
    mov [rbx + TK_TMPL], rax
    movzx ecx, word ptr [rsi + 5]      // syslen
    add rsi, 7
    lea rax, [rsi + rcx]
    cmp rax, rdi
    ja .Lld_bad
    mov [rbx + TK_SYSPTR], rsi
    mov [rbx + TK_SYSLEN], rcx
    add rsi, rcx
    NEED 4
    mov eax, dword ptr [rsi]           // n_special
    add rsi, 4
    cmp eax, 100000
    ja .Lld_bad
    mov [rbx + TK_NSPECIAL], rax
    // SpEnt array (at least one entry so the pointer is never 0)
    lea r8, [rax*8 + 8]
    add r8, r8                         // (n+1)*16
    mov rcx, r13
    xor edx, edx
    call qword ptr [rip + __imp_HeapAlloc]
    test rax, rax
    jz .Lld_fail
    mov [rbx + TK_SPECIALS], rax
    mov r8, rax                        // r8 = &spec[i]
    xor r9d, r9d                       // r9 = i
.Lld_sp:
    cmp r9, [rbx + TK_NSPECIAL]
    jae .Lld_v3
    NEED 7
    mov eax, dword ptr [rsi]           // id
    mov [r8 + SP_ID], eax
    movzx eax, byte ptr [rsi + 4]      // eog flag
    mov [r8 + SP_EOG], al
    movzx ecx, word ptr [rsi + 5]      // literal length
    mov [r8 + SP_LEN], cx
    add rsi, 7
    lea rax, [rsi + rcx]
    cmp rax, rdi
    ja .Lld_bad
    mov [r8 + SP_PTR], rsi
    add rsi, rcx
    add r8, 16
    inc r9
    jmp .Lld_sp

    // ---- v3 tail: the pretokenizer kind ----
.Lld_v3:
    cmp r12d, 3
    jb .Lld_hash
    NEED 1
    movzx eax, byte ptr [rsi]
    cmp eax, 2                         // HFTOK_PRE_MAX
    ja .Lld_bad
    mov [rbx + TK_PRETOK], rax

    // ---- bytes -> id hash map: capacity = smallest power of two >= 2n ----
.Lld_hash:
    mov rcx, [rbx + TK_NVOCAB]
    add rcx, rcx                       // 2n
    mov eax, 1
1:  cmp rax, rcx
    jae 2f
    add rax, rax
    jmp 1b
2:  lea rdx, [rax - 1]
    mov [rbx + TK_HMASK], rdx
    mov r15, rax                       // capacity
    mov rcx, r13
    xor edx, edx
    lea r8, [rax*8]                    // 8-byte slots
    call qword ptr [rip + __imp_HeapAlloc]
    test rax, rax
    jz .Lld_fail
    mov [rbx + TK_HMAP], rax
    mov rdi, rax                       // fill all slots with -1 (rep stosq: [rdi++] = rax, rcx times)
    mov rcx, r15
    mov rax, -1
    rep stosq
    // insert ids in increasing order with linear probing (exactly as the C code,
    // so a lookup finds the LOWEST id among tokens with identical bytes)
    mov r8, [rbx + TK_TOKS]
    mov r10, [rbx + TK_HMAP]
    mov r11, [rbx + TK_HMASK]
    xor r9d, r9d                       // id
.Lld_ins:
    cmp r9, [rbx + TK_NVOCAB]
    jae .Lld_ins_done
    mov rax, r9
    shl rax, 4
    cmp dword ptr [r8 + rax + TE_LEN], 0
    je 2f                              // empty tokens are not hashed
    mov ecx, [r8 + rax + TE_FNV]
    mov eax, ecx
    and rax, r11                       // slot
1:  cmp dword ptr [r10 + rax*8], -1
    je 3f
    inc rax
    and rax, r11
    jmp 1b
3:  mov [r10 + rax*8], r9d             // id
    mov [r10 + rax*8 + 4], ecx         // and its hash
2:  inc r9
    jmp .Lld_ins
.Lld_ins_done:

    // ---- Unicode class table: one byte per code point ----
    // VirtualAlloc returns zeroed pages, so only the classified ranges are written.
    xor ecx, ecx
    mov edx, UCLASS_SIZE
    mov r8d, MEM_COMMIT_RESERVE
    mov r9d, PAGE_READWRITE
    call qword ptr [rip + __imp_VirtualAlloc]
    test rax, rax
    jz .Lld_fail
    mov [rbx + TK_UCLASS], rax
    mov r8, rax
    lea rsi, [rip + tok_unitab]        // rows {lo, hi, bits}, bits = 0 ends the table
1:  mov eax, dword ptr [rsi + 8]       // bits
    test eax, eax
    jz 2f
    mov edx, dword ptr [rsi]           // lo
    mov ecx, dword ptr [rsi + 4]       // hi
    sub ecx, edx
    inc ecx                            // count = hi - lo + 1
    lea rdi, [r8 + rdx]
    rep stosb                          // memset(table + lo, bits, count)
    add rsi, 12
    jmp 1b
2:
    // ---- word cache (zeroed pages = all slots empty) ----
    xor ecx, ecx
    mov edx, WC_ENTRIES << WC_SHIFT
    mov r8d, MEM_COMMIT_RESERVE
    mov r9d, PAGE_READWRITE
    call qword ptr [rip + __imp_VirtualAlloc]
    test rax, rax
    jz .Lld_fail
    mov [rbx + TK_WCACHE], rax
    // ---- pair cache (zeroed = empty) ----
    xor ecx, ecx
    mov edx, PC_ENTRIES * 16
    mov r8d, MEM_COMMIT_RESERVE
    mov r9d, PAGE_READWRITE
    call qword ptr [rip + __imp_VirtualAlloc]
    test rax, rax
    jz .Lld_fail
    mov [rbx + TK_PCACHE], rax

    mov rax, rbx                       // success: return the Tok*
    jmp .Lld_ret

.Lld_bad:                              // malformed file
.Lld_fail:                             // any error: release everything, return 0
    cmp r14, -1
    je 1f
    mov rcx, r14
    call qword ptr [rip + __imp_CloseHandle]
1:  mov rcx, rbx
    call tok_free                      // tok_free(0) is a no-op
    xor eax, eax
.Lld_ret:
    add rsp, 88
    pop r15
    pop r14
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    pop rbp
    ret

// ============================================================================
// void tok_free(Tok* t)  -- frees the buffers and the struct (t may be 0)
// ============================================================================
    .globl tok_free
    .p2align 4
tok_free:
    push rbx
    push rsi
    sub rsp, 40                        // 2 pushes + ret = 24 -> +40 = 64: aligned, 32 shadow
    mov rbx, rcx
    test rbx, rbx
    jz 9f
    mov rcx, [rbx + TK_UCLASS]
    test rcx, rcx
    jz 1f
    xor edx, edx
    mov r8d, MEM_RELEASE
    call qword ptr [rip + __imp_VirtualFree]
1:  mov rcx, [rbx + TK_WCACHE]
    test rcx, rcx
    jz 1f
    xor edx, edx
    mov r8d, MEM_RELEASE
    call qword ptr [rip + __imp_VirtualFree]
1:  mov rcx, [rbx + TK_PCACHE]
    test rcx, rcx
    jz 1f
    xor edx, edx
    mov r8d, MEM_RELEASE
    call qword ptr [rip + __imp_VirtualFree]
1:  // HeapFree each non-null pointer among FILEBUF, TOKS, HMAP, SPECIALS
    lea rsi, [rip + .Lfree_list]
2:  movzx eax, byte ptr [rsi]
    cmp eax, 0xFF
    je 3f
    mov r8, [rbx + rax]
    test r8, r8
    jz 4f
    mov rcx, [rbx + TK_HEAP]
    xor edx, edx
    call qword ptr [rip + __imp_HeapFree]
4:  inc rsi
    jmp 2b
3:  mov rcx, [rbx + TK_HEAP]
    xor edx, edx
    mov r8, rbx
    call qword ptr [rip + __imp_HeapFree]
9:  add rsp, 40
    pop rsi
    pop rbx
    ret

    .section .rdata,"dr"
.Lfree_list:
    .byte TK_FILEBUF, TK_TOKS, TK_HMAP, TK_SPECIALS, 0xFF
    .text

// ============================================================================
// small accessors
// ============================================================================
    .globl tok_vocab
    .p2align 4
tok_vocab:                             // i64 tok_vocab(Tok*)
    mov rax, [rcx + TK_NVOCAB]
    ret

    .globl tok_max_piece
    .p2align 4
tok_max_piece:                         // i64 tok_max_piece(Tok*): longest token, bytes
    mov rax, [rcx + TK_MAXLEN]
    ret

// i64 tok_is_eog(Tok* t, i64 id): id == eos, or a special with that id flagged eog
    .globl tok_is_eog
    .p2align 4
tok_is_eog:
    cmp rdx, [rcx + TK_EOS]
    je 2f
    mov r8, [rcx + TK_NSPECIAL]
    mov r9, [rcx + TK_SPECIALS]
1:  test r8, r8
    jz 3f
    movsxd rax, dword ptr [r9 + SP_ID]
    cmp rax, rdx
    jne 4f
    cmp byte ptr [r9 + SP_EOG], 0
    jne 2f
4:  add r9, 16
    dec r8
    jmp 1b
2:  mov eax, 1
    ret
3:  xor eax, eax
    ret

// i64 tok_piece(Tok* t, i64 id, u8* out): copy the bytes of one token to out and
// return their count (0 for an id outside [0, n_vocab)). Same bytes as
// hftok_decode(t, &id, 1, ...). A token can be up to 65535 bytes long in the file
// format; tok_max_piece() tells the real maximum of a loaded vocab.
    .globl tok_piece
    .p2align 4
tok_piece:
    xor eax, eax
    test rdx, rdx
    js 1f                              // id < 0
    cmp rdx, [rcx + TK_NVOCAB]
    jae 1f                             // id >= n_vocab
    push rsi
    push rdi
    mov rax, [rcx + TK_TOKS]
    shl rdx, 4
    add rax, rdx                       // &toks[id]
    mov rsi, [rax + TE_PTR]
    mov rdi, r8
    mov ecx, [rax + TE_LEN]
    mov eax, ecx                       // return value
    rep movsb                          // memcpy(out, bytes, len)
    pop rdi
    pop rsi
1:  ret

// ============================================================================
// ENCODER
// ============================================================================
// tok_encode and its helpers share a fixed register assignment ("encoder state"),
// so the inner helpers need no arguments for it:
//   rbx = Tok*          rsi = text base         r15 = Unicode class table
//   r12 = ids (out)     r13 = n (ids written)   r14 = cap (max ids)
//   rbp = tok_encode's frame; its locals are addressed as [rbp + E_*]
// Positions are absolute byte offsets into the text.
#define E_LEN      -64    // text length
#define E_PIECES   -72    // i32 pieces[len+1] (BPE work array)
#define E_RANKS    -80    // i32 ranks[len+1]  (rank of pair (pieces[i], pieces[i+1]))
#define E_KIND     -88    // pretokenizer kind
#define E_DROP     -96    // 1 = drop bytes without a token (kind 2), 0 = use id 0
#define E_RUNSTART -104   // start of the current plain run (no specials in it)
#define E_CEND     -112   // end of the current pretoken (inside tok_run)
#define E_WENT     -120   // word-cache entry to fill after this pretoken (0 = none)
#define E_WHASH    -128   // its key hash
#define E_N0       -136   // n before this pretoken
#define E_CSTART   -144   // start of this pretoken

// i64 tok_encode(Tok* t, const u8* text, i64 len, i32* ids, i64 cap)
//
// Exactly hftok_encode(t, text, ids, cap) when len == strlen(text):
//   pos scans the text; at each '<' the LONGEST special literal matching there
//   (first one among equal lengths) is emitted as its id, after the plain run
//   before it has been encoded. Plain runs are split into pretokens by the
//   kind's regex emulation, each pretoken is mapped byte -> single-byte token and
//   merged by BPE ("merge the adjacent pair whose concatenation has the lowest
//   id"). At most cap ids are written; the count is returned.
    .globl tok_encode
    .p2align 4
tok_encode:
    push rbp
    mov rbp, rsp
    push rbx                           // [rbp-8]
    push rsi                           // [rbp-16]
    push rdi                           // [rbp-24]
    push r12                           // [rbp-32]
    push r13                           // [rbp-40]
    push r14                           // [rbp-48]
    push r15                           // [rbp-56]
    sub rsp, 120                       // locals [rbp-144 .. rbp-64]; rsp = rbp-176 (aligned)

    mov rbx, rcx
    mov rsi, rdx
    test r8, r8
    jns 1f
    xor r8d, r8d                       // negative length -> empty text
1:  mov [rbp + E_LEN], r8
    mov r12, r9
    mov r14, [rbp + 48]                // 5th argument: cap (above ret addr + shadow + rbp)
    xor r13d, r13d
    mov r15, [rbx + TK_UCLASS]
    mov rax, [rbx + TK_PRETOK]
    mov [rbp + E_KIND], rax
    xor ecx, ecx
    cmp rax, 2
    sete cl
    mov [rbp + E_DROP], rcx

    // scratch: pieces[len+1] and ranks[len+1] (a pretoken has at most len bytes)
    mov rcx, [rbx + TK_HEAP]
    xor edx, edx
    lea r8, [r8*8 + 8]
    call qword ptr [rip + __imp_HeapAlloc]
    test rax, rax
    jz .Lenc_nomem
    mov [rbp + E_PIECES], rax
    mov rcx, [rbp + E_LEN]
    lea rax, [rax + rcx*4 + 4]
    mov [rbp + E_RANKS], rax

    // ---- main scan for special tokens ----
    xor edi, edi                       // rdi = pos
    mov qword ptr [rbp + E_RUNSTART], 0
.Lenc_scan:
    cmp rdi, [rbp + E_LEN]
    jae .Lenc_tail
    cmp r13, r14
    jge .Lenc_tail
    cmp byte ptr [rsi + rdi], CH_LT
    je .Lenc_lt
    inc rdi
    jmp .Lenc_scan

.Lenc_lt:
    // longest special whose literal matches at pos: best id in eax (-1), length in edx
    mov eax, -1
    xor edx, edx
    mov r8, [rbx + TK_SPECIALS]
    mov r9, [rbx + TK_NSPECIAL]
.Lsp_next:
    test r9, r9
    jz .Lsp_done
    movzx ecx, word ptr [r8 + SP_LEN]  // sl
    cmp ecx, edx
    jbe .Lsp_skip                      // need sl > best_len
    lea r10, [rdi + rcx]
    cmp r10, [rbp + E_LEN]
    ja .Lsp_skip                       // need pos + sl <= len
    // compare text[pos .. pos+sl) with the literal
    mov r10, [r8 + SP_PTR]
    lea r11, [rsi + rdi]
    push rcx
1:  test ecx, ecx
    jz 2f
    dec ecx
    push rax
    movzx eax, byte ptr [r10 + rcx]
    cmp al, byte ptr [r11 + rcx]
    pop rax
    je 1b
    pop rcx
    jmp .Lsp_skip                      // mismatch
2:  pop rcx
    mov eax, [r8 + SP_ID]              // new best (id may even be negative: C does the same)
    mov edx, ecx
.Lsp_skip:
    add r8, 16
    dec r9
    jmp .Lsp_next
.Lsp_done:
    test eax, eax
    js .Lsp_none                       // no match (or best match has id < 0): plain '<'
    // emit the run before the special, then the special itself
    push rax
    push rdx
    mov rcx, [rbp + E_RUNSTART]
    mov rdx, rdi
    call tok_run
    pop rdx
    pop rax
    cmp r13, r14
    jge 3f
    mov [r12 + r13*4], eax
    inc r13
3:  add rdi, rdx                       // skip the literal
    mov [rbp + E_RUNSTART], rdi
    jmp .Lenc_scan
.Lsp_none:
    inc rdi
    jmp .Lenc_scan

.Lenc_tail:
    mov rcx, [rbp + E_RUNSTART]
    mov rdx, [rbp + E_LEN]
    call tok_run

    mov rcx, [rbx + TK_HEAP]
    xor edx, edx
    mov r8, [rbp + E_PIECES]
    call qword ptr [rip + __imp_HeapFree]
    mov rax, r13
    jmp .Lenc_ret
.Lenc_nomem:
    xor eax, eax
.Lenc_ret:
    add rsp, 120
    pop r15
    pop r14
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    pop rbp
    ret

// ----------------------------------------------------------------------------
// tok_run: encode the plain text [rcx, rdx) (C: encode_run).
// In: rcx = start, rdx = end + the encoder state. Out: r13 updated.
// Preserves rdi; clobbers rax rcx rdx r8-r11.
//   r8 = run end, r9 = position, rdi = SmolLM2 segment end (state of kind 2),
//   r10 = number of pieces, r11 = pieces array
// ----------------------------------------------------------------------------
    .p2align 4
tok_run:
    push rdi
    mov r8, rdx
    mov r9, rcx
    mov rdi, rcx                       // seg_end = start (C: seg = 0 relative to the run)
.Lrun_chunk:
    cmp r13, r14
    jge .Lrun_ret                      // ids full
    cmp r9, r8
    jge .Lrun_ret                      // run consumed
    // next pretoken [r9, rax)
    mov rax, [rbp + E_KIND]
    cmp eax, 1
    je 1f
    cmp eax, 2
    je 2f
    call tok_pre_gpt2
    jmp 3f
1:  call tok_pre_qwen2
    jmp 3f
2:  call tok_pre_smol
3:  mov [rbp + E_CEND], rax
    mov qword ptr [rbp + E_WENT], 0
    mov rcx, rax
    sub rcx, r9                        // rcx = pretoken length
    cmp rcx, 1
    je .Lrun_one
    cmp rcx, WC_KEYMAX
    ja .Lrun_bytes                     // too long for the cache
    // ---- word cache lookup: FNV-1a of the bytes picks the slot ----
    lea r11, [rsi + r9]                // r11 = the pretoken bytes
    mov r10d, FNV_BASIS
    xor edx, edx
1:  movzx eax, byte ptr [r11 + rdx]
    xor r10d, eax
    imul r10d, r10d, FNV_PRIME
    inc edx
    cmp edx, ecx
    jb 1b
    mov [rbp + E_WHASH], r10
    mov eax, r10d
    and eax, WC_ENTRIES - 1
    shl rax, WC_SHIFT
    add rax, [rbx + TK_WCACHE]         // rax = &entry
    cmp [rax + WC_HASH], r10d
    jne .Lrun_miss
    cmp [rax + WC_KLEN], cx
    jne .Lrun_miss
    xor edx, edx                       // same hash + length: compare the bytes,
5:  lea r10d, [rdx + 8]                // 8 at a time while 8 are left ...
    cmp r10d, ecx
    ja 2f
    mov r10, [r11 + rdx]
    cmp r10, [rax + WC_KEY + rdx]
    jne .Lrun_miss
    add edx, 8
    jmp 5b
2:  cmp edx, ecx                       // ... then one at a time
    jae 6f
    movzx r10d, byte ptr [r11 + rdx]
    cmp r10b, [rax + WC_KEY + rdx]
    jne .Lrun_miss
    inc edx
    jmp 2b
6:
    // hit: append the cached ids (while n < cap)
    mov r11, rax
    movzx ecx, word ptr [r11 + WC_NIDS]
    xor edx, edx
3:  cmp edx, ecx
    jae 4f
    cmp r13, r14
    jge 4f
    mov eax, [r11 + WC_IDS + rdx*4]
    mov [r12 + r13*4], eax
    inc r13
    inc edx
    jmp 3b
4:  mov r9, [rbp + E_CEND]
    jmp .Lrun_chunk
.Lrun_miss:                            // encode normally, then fill this entry
    mov [rbp + E_WENT], rax
    mov [rbp + E_N0], r13
    mov [rbp + E_CSTART], r9
    jmp .Lrun_bytes

.Lrun_one:
    // fast path: a one-byte pretoken that has a single-byte token is that token
    movzx ecx, byte ptr [rsi + r9]
    mov ecx, [rbx + TK_B2ID + rcx*4]
    test ecx, ecx
    js .Lrun_bytes
    mov [r12 + r13*4], ecx             // n < cap was checked above
    inc r13
    mov r9, rax
    jmp .Lrun_chunk

.Lrun_bytes:
    // bytes -> pieces (C: the for (i = 0; i <= cl; i++) loop)
    mov r11, [rbp + E_PIECES]
    xor r10d, r10d
.Lrun_byte:
    cmp r9, [rbp + E_CEND]
    jae .Lrun_flush_end
    movzx eax, byte ptr [rsi + r9]
    inc r9
    mov eax, [rbx + TK_B2ID + rax*4]
    test eax, eax
    jns .Lrun_put
    cmp qword ptr [rbp + E_DROP], 0
    jne .Lrun_drop
    xor eax, eax                       // kinds 0/1: missing byte token -> id 0
.Lrun_put:
    mov [r11 + r10*4], eax
    inc r10
    jmp .Lrun_byte
.Lrun_drop:                            // kind 2: the byte is dropped and is a merge barrier
    call tok_flush
    jmp .Lrun_byte
.Lrun_flush_end:
    call tok_flush
    // fill the word-cache entry chosen above (only if the output was not cut by cap)
    mov r11, [rbp + E_WENT]
    test r11, r11
    jz .Lrun_chunk
    cmp r13, r14
    jge .Lrun_chunk
    mov rax, r13
    sub rax, [rbp + E_N0]
    cmp rax, WC_IDSMAX
    ja .Lrun_chunk                     // too many ids to cache
    mov eax, [rbp + E_WHASH]
    mov [r11 + WC_HASH], eax
    mov rdx, [rbp + E_CSTART]
    mov rcx, [rbp + E_CEND]
    sub rcx, rdx                       // key length
    mov [r11 + WC_KLEN], cx
    xor eax, eax                       // copy the key bytes
1:  mov r10b, [rsi + rdx]
    mov [r11 + WC_KEY + rax], r10b
    inc rdx
    inc eax
    cmp eax, ecx
    jb 1b
    mov rdx, [rbp + E_N0]              // copy the ids produced: ids[n0 .. n)
    mov rcx, r13
    sub rcx, rdx
    mov [r11 + WC_NIDS], cx
    lea rdx, [r12 + rdx*4]
    xor eax, eax
2:  cmp eax, ecx
    jae .Lrun_chunk
    mov r10d, [rdx + rax*4]
    mov [r11 + WC_IDS + rax*4], r10d
    inc eax
    jmp 2b
.Lrun_ret:
    pop rdi
    ret

// tok_flush: BPE-merge pieces[0..r10) and append them to ids (while n < cap).
// Resets r10 to 0. Preserves r8 r9 rdi r11 (and the encoder state).
    .p2align 4
tok_flush:
    cmp r10, 1
    jbe 1f
    call tok_bpe                       // r10 = merged count
1:  xor ecx, ecx
2:  cmp rcx, r10
    jae 3f
    cmp r13, r14
    jge 3f
    mov eax, [r11 + rcx*4]
    mov [r12 + r13*4], eax
    inc r13
    inc rcx
    jmp 2b
3:  xor r10d, r10d
    ret

// ----------------------------------------------------------------------------
// tok_bpe: merge pieces[0..r10) (r11 = pieces) in place; returns the new count in r10.
// Rule (same result as the C loop): find the pair with the LOWEST merged id,
// leftmost on ties; replace it by the merged id; repeat until no pair merges.
// ranks[i] caches the merged id of (pieces[i], pieces[i+1]) or NO_RANK, so each
// round only the two pairs touching the merge are looked up again.
// Preserves everything except rax rcx rdx r10.
// ----------------------------------------------------------------------------
    .p2align 4
tok_bpe:
    push r8
    push r9
    push rdi
    push r12
    push r13
    push r14
    push r15
    push r11
    mov r12, r11                       // r12 = pieces
    mov r13, [rbp + E_RANKS]           // r13 = ranks
    mov r14, r10                       // r14 = count
    // initial ranks for every adjacent pair
    xor r15d, r15d
1:  lea rax, [r15 + 1]
    cmp rax, r14
    jae .Lbpe_round
    mov ecx, [r12 + r15*4]
    mov edx, [r12 + r15*4 + 4]
    call tok_pairrank
    mov [r13 + r15*4], eax
    inc r15
    jmp 1b

.Lbpe_round:
    // argmin over ranks[0 .. count-2] (strict < keeps the leftmost)
    mov eax, NO_RANK                   // best rank
    mov r15, -1                        // best index
    xor ecx, ecx
    lea rdx, [r14 - 1]                 // number of pairs
2:  cmp rcx, rdx
    jae 3f
    cmp [r13 + rcx*4], eax
    jae 4f
    mov eax, [r13 + rcx*4]
    mov r15, rcx
4:  inc rcx
    jmp 2b
3:  test r15, r15
    js .Lbpe_done                      // nothing merges any more
    // merge: pieces[i] = rank; delete pieces[i+1] and ranks[i+1] (shift left)
    mov [r12 + r15*4], eax
    lea rcx, [r15 + 1]
    lea rdx, [r14 - 1]                 // last index (old count - 1)
5:  cmp rcx, rdx
    jae 6f
    mov eax, [r12 + rcx*4 + 4]
    mov [r12 + rcx*4], eax
    mov eax, [r13 + rcx*4 + 4]         // ranks[j] = ranks[j+1] (the last copy reads a
    mov [r13 + rcx*4], eax             // stale slot that lies outside the valid range)
    inc rcx
    jmp 5b
6:  dec r14
    // refresh the pair on the left (i-1, i) ...
    test r15, r15
    jz 7f
    mov ecx, [r12 + r15*4 - 4]
    mov edx, [r12 + r15*4]
    call tok_pairrank
    mov [r13 + r15*4 - 4], eax
7:  // ... and on the right (i, i+1)
    lea rax, [r15 + 1]
    cmp rax, r14
    jae .Lbpe_round
    mov ecx, [r12 + r15*4]
    mov edx, [r12 + r15*4 + 4]
    call tok_pairrank
    mov [r13 + r15*4], eax
    jmp .Lbpe_round

.Lbpe_done:
    mov r10, r14
    pop r11
    pop r15
    pop r14
    pop r13
    pop r12
    pop rdi
    pop r9
    pop r8
    ret

// ----------------------------------------------------------------------------
// tok_pairrank: id of the token whose bytes are bytes(a) || bytes(b), or NO_RANK.
// In: ecx = a, edx = b (both >= 0, upper 32 bits zero), rbx = Tok*. Out: eax.
// Clobbers rax rcx rdx r10 r11 (tok_bpe, the only caller, keeps nothing there).
// First looks in the pair cache; on a miss does the real lookup and stores it.
// ----------------------------------------------------------------------------
    .p2align 4
tok_pairrank:
    mov eax, ecx
    shl rax, 32
    or rax, rdx
    bts rax, 63                        // key = a<<32 | b | tag bit
    mov r10, 0x9E3779B97F4A7C15        // 2^64 / golden ratio (Fibonacci hashing)
    imul r10, rax
    shr r10, 64 - PC_BITS              // top bits of the product = slot
    shl r10, 4
    add r10, [rbx + TK_PCACHE]         // r10 = &entry
    cmp [r10], rax
    jne 1f
    mov eax, [r10 + 8]                 // hit
    ret
1:  push r10
    push rax
    call tok_pairrank_slow
    pop rcx
    pop r10
    mov [r10], rcx                     // remember it
    mov [r10 + 8], eax
    ret

// tok_pairrank_slow: the real hash-map lookup (C: memcpy both into buf, then
// hftok_lookup(buf)). In: ecx = a, edx = b. Out: eax. Clobbers rax rcx rdx only.
// A hash slot holds {i32 id, u32 full FNV hash}: a slot whose hash differs is
// skipped without touching the token table (fewer cache misses); the bytes are
// still compared when the hash matches, so the result is exact.
    .p2align 4
tok_pairrank_slow:
    push rsi
    push rdi
    push r8
    push r9
    push r10
    push r11
    push r12
    push r13
    mov r8, [rbx + TK_TOKS]
    shl rcx, 4
    shl rdx, 4
    add rcx, r8                        // &toks[a]
    add rdx, r8                        // &toks[b]
    mov r9, [rcx + TE_PTR]             // r9  = bytes(a)
    mov r10d, [rcx + TE_LEN]           // r10 = len(a)
    mov rsi, [rdx + TE_PTR]            // rsi = bytes(b)
    mov edi, [rdx + TE_LEN]            // rdi = len(b)
    // hash(a||b): continue FNV-1a from hash(a) over the bytes of b
    mov eax, [rcx + TE_FNV]
    xor ecx, ecx
1:  cmp ecx, edi
    jae 2f
    movzx edx, byte ptr [rsi + rcx]
    xor eax, edx
    imul eax, eax, FNV_PRIME
    inc ecx
    jmp 1b
2:  mov r13d, eax                      // r13 = full hash
    mov r12, [rbx + TK_HMAP]
    and rax, [rbx + TK_HMASK]          // slot
.Lpr_probe:
    movsxd rdx, dword ptr [r12 + rax*8]   // id in this slot
    test rdx, rdx
    js .Lpr_miss                       // empty slot: not in the vocab
    cmp [r12 + rax*8 + 4], r13d
    jne .Lpr_next                      // different hash: different bytes
    mov rcx, rdx
    shl rcx, 4
    add rcx, r8                        // &toks[id]
    lea r11d, [r10 + rdi]              // total length must match
    cmp [rcx + TE_LEN], r11d
    jne .Lpr_next
    mov rcx, [rcx + TE_PTR]            // candidate bytes
    // compare the first len(a) bytes with bytes(a) ...
    xor r11d, r11d
3:  cmp r11d, r10d
    jae 4f
    push rax
    movzx eax, byte ptr [r9 + r11]
    cmp al, byte ptr [rcx + r11]
    pop rax
    jne .Lpr_next
    inc r11d
    jmp 3b
4:  // ... then the remaining len(b) bytes with bytes(b)
    add rcx, r10
    xor r11d, r11d
5:  cmp r11d, edi
    jae 7f                             // all equal: found
    push rax
    movzx eax, byte ptr [rsi + r11]
    cmp al, byte ptr [rcx + r11]
    pop rax
    jne .Lpr_next
    inc r11d
    jmp 5b
.Lpr_next:
    inc rax
    and rax, [rbx + TK_HMASK]
    jmp .Lpr_probe
7:  mov eax, edx                       // the id
    jmp .Lpr_ret
.Lpr_miss:
    mov eax, NO_RANK
.Lpr_ret:
    pop r13
    pop r12
    pop r11
    pop r10
    pop r9
    pop r8
    pop rdi
    pop rsi
    ret

// ============================================================================
// PRETOKENIZERS
// ============================================================================
// Common contract: in  r9 = p (start of the pretoken, p < r8), r8 = end of the
//                       text the regex sees, rsi = text, r15 = class table
//                  out rax = end of the pretoken (its start is always p)
// They preserve every register except rax rcx rdx r10 r11.
// (C: next_chunk / next_chunk_qwen2 / next_chunk_smol with *cs = p.)

// ----------------------------------------------------------------------------
// tok_dec: decode the UTF-8 char at rcx (rcx < r8).
// Out: eax = code point, edx = its byte length, r10d = class bits.
// A malformed / truncated / overlong / surrogate / > U+10FFFF sequence is ONE
// byte with code point 0xFFFFFFFF and class 0 ("other"), as in C's u_decode.
// Preserves everything except rax rdx r10 r11.
// ----------------------------------------------------------------------------
.macro CONT off                        // fold continuation byte [rsi+rcx+off] into eax
    movzx edx, byte ptr [rsi + rcx + \off]
    mov r10d, edx
    and r10d, 0xC0
    cmp r10d, 0x80                     // must look like 10xxxxxx
    jne .Ldec_bad
    shl eax, 6
    and edx, 0x3F
    or eax, edx
.endm

    .p2align 4
tok_dec:
    movzx eax, byte ptr [rsi + rcx]
    cmp eax, 0x80
    jae .Ldec_multi
    mov edx, 1                         // ASCII: one byte, class from the table
    movzx r10d, byte ptr [r15 + rax]
    ret
.Ldec_multi:
    mov r11d, eax
    and r11d, 0xE0
    cmp r11d, 0xC0
    je .Ldec_2                         // 110xxxxx
    mov r11d, eax
    and r11d, 0xF0
    cmp r11d, 0xE0
    je .Ldec_3                         // 1110xxxx
    mov r11d, eax
    and r11d, 0xF8
    cmp r11d, 0xF0
    je .Ldec_4                         // 11110xxx
    jmp .Ldec_bad                      // continuation byte or 0xF8..0xFF as a lead
.Ldec_2:
    lea r11, [rcx + 2]
    cmp r11, r8
    ja .Ldec_bad                       // truncated
    and eax, 0x1F
    CONT 1
    cmp eax, 0x80
    jb .Ldec_bad                       // overlong
    mov edx, 2
    jmp .Ldec_cls
.Ldec_3:
    lea r11, [rcx + 3]
    cmp r11, r8
    ja .Ldec_bad
    and eax, 0x0F
    CONT 1
    CONT 2
    cmp eax, 0x800
    jb .Ldec_bad
    mov r11d, eax
    and r11d, 0xFFFFF800
    cmp r11d, 0xD800                   // U+D800..U+DFFF are surrogates
    je .Ldec_bad
    mov edx, 3
    jmp .Ldec_cls
.Ldec_4:
    lea r11, [rcx + 4]
    cmp r11, r8
    ja .Ldec_bad
    and eax, 0x07
    CONT 1
    CONT 2
    CONT 3
    cmp eax, 0x10000
    jb .Ldec_bad
    cmp eax, 0x10FFFF
    ja .Ldec_bad
    mov edx, 4
.Ldec_cls:
    movzx r10d, byte ptr [r15 + rax]
    ret
.Ldec_bad:
    mov eax, 0xFFFFFFFF
    mov edx, 1
    xor r10d, r10d
    ret

// ----------------------------------------------------------------------------
// Kind 0: GPT-2 regex with ASCII-approximate classes, on BYTES (C: next_chunk)
//   's|'t|'re|'ve|'m|'ll|'d | ?\p{L}+ | ?\p{N}+ | ?[^\s\p{L}\p{N}]+ | \s+(?!\S) | \s+
// where a "letter" is a-z, A-Z or ANY byte >= 0x80.
// CLS0: edx = class of byte eax under these rules.
// ----------------------------------------------------------------------------
.macro CLS0
    mov edx, C_L
    cmp eax, 0x80
    jae 91f
    movzx edx, byte ptr [r15 + rax]
91:
.endm

    .p2align 4
tok_pre_gpt2:
    movzx eax, byte ptr [rsi + r9]
    // contractions: ' followed by s t m d (2 bytes) or re ve ll (3 bytes)
    cmp eax, CH_QUOTE
    jne .Lg0_word
    lea rcx, [r9 + 1]
    cmp rcx, r8
    jae .Lg0_word
    movzx edx, byte ptr [rsi + r9 + 1]
    cmp edx, CH_s
    je .Lg0_ret2
    cmp edx, CH_t
    je .Lg0_ret2
    cmp edx, CH_m
    je .Lg0_ret2
    cmp edx, CH_d
    je .Lg0_ret2
    lea rcx, [r9 + 2]
    cmp rcx, r8
    jae .Lg0_word
    movzx ecx, byte ptr [rsi + r9 + 2]
    cmp edx, CH_r
    je 1f
    cmp edx, CH_v
    je 1f
    cmp edx, CH_l
    jne .Lg0_word
    cmp ecx, CH_l                      // 'll
    je .Lg0_ret3
    jmp .Lg0_word
1:  cmp ecx, CH_e                      // 're / 've
    je .Lg0_ret3
    jmp .Lg0_word
.Lg0_ret2:
    lea rax, [r9 + 2]
    ret
.Lg0_ret3:
    lea rax, [r9 + 3]
    ret

.Lg0_word:
    // one optional leading space, then a run of letters / digits / "other"
    mov rcx, r9
    cmp eax, CH_SP
    jne 1f
    inc rcx
1:  cmp rcx, r8
    jae .Lg0_ws
    movzx eax, byte ptr [rsi + rcx]
    CLS0
    cmp edx, C_S
    je .Lg0_ws
    mov r10d, edx                      // the class of this run
2:  inc rcx
    cmp rcx, r8
    jae 3f
    movzx eax, byte ptr [rsi + rcx]
    CLS0
    cmp edx, r10d
    je 2b
3:  mov rax, rcx
    ret

.Lg0_ws:
    // whitespace run; if a non-space follows and the run has 2+ bytes, leave
    // its last byte for the next pretoken (\s+(?!\S))
    mov rcx, r9
1:  cmp rcx, r8
    jae 2f
    movzx eax, byte ptr [rsi + rcx]
    CLS0
    cmp edx, C_S
    jne 2f
    inc rcx
    jmp 1b
2:  mov rax, rcx
    cmp rcx, r8
    jae 3f                             // run reaches the end: take it all
    lea rdx, [rcx - 1]
    cmp rdx, r9
    jbe 3f                             // only one space: take it
    mov rax, rdx
3:  ret

// ----------------------------------------------------------------------------
// Kind 1: Qwen2 / Qwen2.5 / Qwen3 regex with real Unicode classes (C: next_chunk_qwen2)
//   (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?\p{L}+ | \p{N}
//   | ?[^\s\p{L}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
// The first alternative that matches at p wins.
// Locals: r12d = c0, r13 = n0, r14d = class(c0), ebx = c1 (0xFFFFFFFE = end of
// text), rdi = n1, [rsp] = class(c1), [rsp+4] = case-folded c1.
// ----------------------------------------------------------------------------
#define END_CP 0xFFFFFFFE
// FOLD reg: case-fold a code point for the contraction test: U+017F (long s) -> 's',
// otherwise OR 0x20 (maps A-Z to a-z; never turns a non-letter into a-z)
.macro FOLD reg
    cmp \reg, 0x17F
    jne 92f
    mov \reg, CH_s
    jmp 93f
92: or \reg, 0x20
93:
.endm

    .p2align 4
tok_pre_qwen2:
    push rbx
    push rdi
    push r12
    push r13
    push r14
    sub rsp, 16
    mov rcx, r9
    call tok_dec
    mov r12d, eax
    mov r13d, edx
    mov r14d, r10d
    mov ebx, END_CP
    xor edi, edi
    mov dword ptr [rsp], 0
    lea rcx, [r9 + r13]
    cmp rcx, r8
    jae 1f
    call tok_dec
    mov ebx, eax
    mov edi, edx
    mov [rsp], r10d
1:
    // ---- 1. case-insensitive contractions ----
    cmp r12d, CH_QUOTE
    jne .Lq_letters
    cmp ebx, END_CP
    je .Lq_letters
    mov eax, ebx
    FOLD eax
    mov [rsp + 4], eax
    cmp eax, CH_s
    je .Lq_ret2
    cmp eax, CH_t
    je .Lq_ret2
    cmp eax, CH_m
    je .Lq_ret2
    cmp eax, CH_d
    je .Lq_ret2
    lea rcx, [r9 + r13]
    add rcx, rdi                       // position of c2
    cmp rcx, r8
    jae .Lq_letters
    call tok_dec                       // eax = c2, edx = n2 (rcx preserved)
    FOLD eax
    mov r10d, [rsp + 4]                // folded c1
    cmp r10d, CH_r
    je 2f
    cmp r10d, CH_v
    je 2f
    cmp r10d, CH_l
    jne .Lq_letters
    cmp eax, CH_l                      // 'll
    je 3f
    jmp .Lq_letters
2:  cmp eax, CH_e                      // 're / 've
    jne .Lq_letters
3:  lea rax, [rcx + rdx]               // end = position of c2 + n2
    jmp .Lq_ret
.Lq_ret2:
    lea rax, [r9 + r13]
    add rax, rdi
    jmp .Lq_ret

    // ---- 2. [^\r\n\p{L}\p{N}]?\p{L}+ ----
.Lq_letters:
    mov rcx, r9                        // q = p if c0 is a letter
    test r14d, C_L
    jnz 4f
    cmp r12d, CH_CR                    // else q = p + n0 if c0 is not CR/LF/number
    je .Lq_number                      // and c1 exists and is a letter
    cmp r12d, CH_LF
    je .Lq_number
    test r14d, C_N
    jnz .Lq_number
    cmp ebx, END_CP
    je .Lq_number
    test dword ptr [rsp], C_L
    jz .Lq_number
    add rcx, r13
4:  // letter run from q
    cmp rcx, r8
    jae 5f
    call tok_dec
    test r10d, C_L
    jz 5f
    add rcx, rdx
    jmp 4b
5:  mov rax, rcx
    jmp .Lq_ret

    // ---- 3. \p{N}: a single code point ----
.Lq_number:
    test r14d, C_N
    jz .Lq_punct
    lea rax, [r9 + r13]
    jmp .Lq_ret

    // ---- 4.  ?[^\s\p{L}\p{N}]+[\r\n]* ----
.Lq_punct:
    mov rcx, r9
    cmp r12d, CH_SP
    jne 1f
    add rcx, r13
1:  mov rdi, rcx                       // r = start of the "other" run
2:  cmp rcx, r8
    jae 3f
    call tok_dec
    test r10d, C_L | C_N | C_S
    jnz 3f
    add rcx, rdx
    jmp 2b
3:  cmp rcx, rdi
    je .Lq_space                       // empty run: not this alternative
4:  cmp rcx, r8                        // swallow trailing CR / LF bytes
    jae 5f
    movzx eax, byte ptr [rsi + rcx]
    cmp eax, CH_CR
    je 6f
    cmp eax, CH_LF
    jne 5f
6:  inc rcx
    jmp 4b
5:  mov rax, rcx
    jmp .Lq_ret

    // ---- 5-7. whitespace (c0 is \s here) ----
.Lq_space:
    mov rcx, r9                        // e
    mov rdi, -1                        // last CR/LF position
    mov rbx, r9                        // start of the last space char
1:  cmp rcx, r8
    jae 2f
    call tok_dec
    test r10d, C_S
    jz 2f
    cmp eax, CH_CR
    je 3f
    cmp eax, CH_LF
    jne 4f
3:  mov rdi, rcx
4:  mov rbx, rcx
    add rcx, rdx
    jmp 1b
2:  cmp rcx, r9
    jne 5f
    lea rax, [r9 + r13]                // unreachable safety net (as in C)
    jmp .Lq_ret
5:  test rdi, rdi
    js 6f
    lea rax, [rdi + 1]                 // 5. \s*[\r\n]+ : up to the LAST CR/LF
    jmp .Lq_ret
6:  cmp rcx, r8
    jae 7f
    cmp rbx, r9
    jbe 7f
    mov rax, rbx                       // 6. \s+(?!\S): leave the last char
    jmp .Lq_ret
7:  mov rax, rcx                       // 6 at end / 7. \s+
.Lq_ret:
    add rsp, 16
    pop r14
    pop r13
    pop r12
    pop rdi
    pop rbx
    ret

// ----------------------------------------------------------------------------
// Kind 2 helper: GPT-2 regex with real Unicode classes on [p, r8) (C: next_chunk_gpt2u)
//   's|'t|'re|'ve|'m|'ll|'d | ?\p{L}+ | ?\p{N}+ | ?[^\s\p{L}\p{N}]+ | \s+(?!\S) | \s+
// Class bits are one of 0 (other), C_L, C_N, C_S, so "same class" = equal bits.
// Locals: r12d = c0, r13 = n0, r14d = class(c0), ebx = c1, edi = class(c1) or -1
// when there is no next char.
// ----------------------------------------------------------------------------
    .p2align 4
tok_pre_gpt2u:
    push rbx
    push rdi
    push r12
    push r13
    push r14
    mov rcx, r9
    call tok_dec
    mov r12d, eax
    mov r13d, edx
    mov r14d, r10d
    xor ebx, ebx
    mov edi, -1
    lea rcx, [r9 + r13]
    cmp rcx, r8
    jae 1f
    call tok_dec
    mov ebx, eax
    mov edi, r10d
1:
    // ---- contractions (case-sensitive) ----
    cmp r12d, CH_QUOTE
    jne .Lgu_word
    cmp edi, -1
    je .Lgu_word
    cmp ebx, CH_s
    je .Lgu_ret2
    cmp ebx, CH_t
    je .Lgu_ret2
    cmp ebx, CH_m
    je .Lgu_ret2
    cmp ebx, CH_d
    je .Lgu_ret2
    lea rcx, [r9 + 2]
    cmp rcx, r8
    jae .Lgu_word
    movzx eax, byte ptr [rsi + r9 + 2]
    cmp ebx, CH_r
    je 2f
    cmp ebx, CH_v
    je 2f
    cmp ebx, CH_l
    jne .Lgu_word
    cmp eax, CH_l
    je .Lgu_ret3
    jmp .Lgu_word
2:  cmp eax, CH_e
    je .Lgu_ret3
    jmp .Lgu_word
.Lgu_ret2:
    lea rax, [r9 + 2]
    jmp .Lgu_ret
.Lgu_ret3:
    lea rax, [r9 + 3]
    jmp .Lgu_ret

.Lgu_word:
    // optional ' ' then a run of one class (letters / numbers / other)
    mov rcx, r9
    mov edx, r14d
    cmp r12d, CH_SP
    jne 3f
    cmp edi, -1
    je 3f
    cmp edi, C_S
    je 3f
    add rcx, r13                       // ' ' + a non-space: the run starts after it
    mov edx, edi
3:  cmp edx, C_S
    je .Lgu_ws
    mov ebx, edx                       // run class
4:  cmp rcx, r8
    jae 5f
    call tok_dec
    cmp r10d, ebx
    jne 5f
    add rcx, rdx
    jmp 4b
5:  mov rax, rcx
    jmp .Lgu_ret

.Lgu_ws:
    mov rcx, r9
    mov rbx, r9                        // start of the last space char
1:  cmp rcx, r8
    jae 2f
    call tok_dec
    cmp r10d, C_S
    jne 2f
    mov rbx, rcx
    add rcx, rdx
    jmp 1b
2:  mov rax, rcx
    cmp rcx, r8
    jae .Lgu_ret
    cmp rbx, r9
    jbe .Lgu_ret
    mov rax, rbx
.Lgu_ret:
    pop r14
    pop r13
    pop r12
    pop rdi
    pop rbx
    ret

// ----------------------------------------------------------------------------
// Kind 2: SmolLM2 exact (C: next_chunk_smol). HF first splits every numeric char
// (\p{N}) into its own piece, then runs the Unicode GPT-2 regex on each other
// piece separately. rdi carries the end of the current non-numeric piece
// between calls (tok_run starts it at the run start).
// ----------------------------------------------------------------------------
    .p2align 4
tok_pre_smol:
    cmp r9, rdi
    jb 3f                              // still inside the current piece
    mov rcx, r9
    call tok_dec                       // (decoded against the full run end)
    test r10d, C_N
    jz 1f
    lea rax, [r9 + rdx]                // a number char is a pretoken on its own
    ret
1:  // find the end of this non-numeric piece
    cmp rcx, r8
    jae 2f
    call tok_dec
    test r10d, C_N
    jnz 2f
    add rcx, rdx
    jmp 1b
2:  mov rdi, rcx
3:  push r8
    mov r8, rdi                        // the piece end acts as end-of-text
    call tok_pre_gpt2u
    pop r8
    ret
