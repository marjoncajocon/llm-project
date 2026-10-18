// decode.S - one-token decode step and batched prompt prefill (owner: decode agent)
//
// Exported (Win64 ABI, ints are int64):
//   i64 decode_init(Model* m)
//        Allocate the scratch buffers for model m (idempotent: a second call with
//        the same model does nothing; a different model frees and reallocates).
//        decode_step/prefill call it themselves. Returns 0, or 1 if out of memory.
//   i64 decode_step(Model* m, KV* kv, i64 token, float* logits)
//        Decode `token` at position kv->len: append its K/V to every layer's ring
//        cache, attend, and (logits != 0) write vocab logits. Advances kv->len.
//        logits == 0 = "prefill" mode: like llama_decode_step_lora with NULL logits
//        it stops after caching the LAST layer's K/V (that layer's attention + FFN,
//        the final norm and the head do not influence the cache).
//        Returns 0, or 1 if the cache is full (nothing done).
//   i64 prefill(Model* m, KV* kv, const i32* ids, i64 n)
//        Push n prompt tokens through the model into the KV cache WITHOUT logits,
//        in batches of up to PF_ROWS rows: every weight matrix is streamed once
//        per batch (matmul_rows) instead of once per token. The cache ends up
//        bit-identical to n decode_step(..., 0) calls (row i of matmul_rows ==
//        matvec of row i, and every per-row op runs the same code). To get the
//        logits of the last prompt token: prefill(ids, n-1) then
//        decode_step(ids[n-1], logits). Returns 0, or 1 if the cache filled up.
//
// The per-token computation (llama.c q_llama_decode_step / llama_decode_step_lora
// with lo = NULL), for every layer l:
//   xn = rmsnorm(x, attn_norm)
//   q = xn@Wq + b_q ; k = xn@Wk + b_k ; v = xn@Wv + b_v
//   [qk_norm: per-head rmsnorm of q (q_norm) and k (k_norm)]
//   RoPE(q), RoPE(k) at position pos ; K/V cache slot pos % cap[l] = k, v
//   o = attention(q, cache) ; x += o@Wo
//   hn = rmsnorm(x, ffn_norm) ; g = hn@Wgate ; u = hn@Wup ; x += (SiLU(g)*u)@Wdown
// then logits = rmsnorm(x, final_norm) @ lm_head.
//
// k and v are computed directly INTO the cache slot (matvec writes there, then
// bias / QK-norm / RoPE work in place) -- no copy.
//
// int8 layers (Wq kind Q8, dec_fuse = 1) issue 4 fused pool jobs per layer
// instead of 7 matvecs + 5 elementwise passes (matvec_q8_multi, matvec.S):
//   1. q, k, v together (+ biases)          2. x += o@Wo
//   3. g = SiLU(xn@Wgate) * (xn@Wup)        4. x += g@Wdown
// Same arithmetic, same order per element, so the result is bit-identical
// (tests/dec_test.c: logits diff 0 vs C, prefill cache == decode cache).
// fp32 layers keep the one-matvec-per-matrix path.
// Prefill runs PF_ROWS = 256 rows per batch, and its rmsnorm / SiLU passes are
// split over the pool by rows (par_rmsnorm_rows / par_silu_rows).
//
// Profiling: with dec_prof_on = 1 every PMARK adds rdtsc ticks to dec_prof[slot]
// (tests/dec_test.c DEC_ONLY=prof prints the per-op table).

#include "model.inc"
        .intel_syntax noprefix

// AttnArgs (identical to ops.S)
#define AT_Q        0
#define AT_KC       8
#define AT_VC       16
#define AT_CAP      24
#define AT_WINDOW   32
#define AT_POS0     40
#define AT_NROWS    48
#define AT_NHEADS   56
#define AT_NKV      64
#define AT_HD       72
#define AT_SC       80
#define AT_SCSTRIDE 88
#define AT_O        96
#define AT_SIZE     128

#define PF_ROWS     256                 // prefill batch (rows per weight pass; 256 measured
                                        // ~25% faster than 64 on a 424-token prompt)
#define PSC_FLOATS  (4 * 1024 * 1024)   // prefill attention score scratch (16 MB)

#define MEM_COMMIT_RESERVE 0x3000
#define MEM_RELEASE        0x8000
#define PAGE_READWRITE     0x04

// ---------------------------------------------------------------------------
// Static decoder context. The buffer pointers form one contiguous table
// (dc_ptrs .. dc_ptrs_end) so decode_init can relocate them in a loop.
// ---------------------------------------------------------------------------
        .bss
        .p2align 6
dc_model:    .space 8           // Model* the buffers were sized for
dc_sig:      .space 80          // copy of Model bytes 0..71 (dims, rope_base, eps)
                                // + the layers pointer: a reloaded / different
                                // model at the same address is detected too
dc_block:    .space 8           // VirtualAlloc base of all buffers
dc_scstride: .space 8           // floats of score scratch per head (decode)
dc_args:     .space AT_SIZE     // AttnArgs reused by every attention call
dc_tasks:    .space 4*32        // MvTask list for matvec_q8_multi (see matvec.S)
dc_ptrs:
dc_x:   .space 8                // residual stream (d_model)
dc_xn:  .space 8                // normed input (d_model), also hn
dc_q:   .space 8                // q (n_heads*hd)
dc_k:   .space 8                // k of this token (kv_dim) before kv_put
dc_o:   .space 8                // attention output (n_heads*hd)
dc_ao:  .space 8                // projection output (d_model), also FFN out
dc_g:   .space 8                // gate (d_ff)
dc_u:   .space 8                // up (d_ff)
dc_cs:  .space 8                // RoPE cos/sin table of the current position (hd)
dc_inv: .space 8                // RoPE inverse frequencies (hd/2)
dc_sc:  .space 8                // attention scores (n_heads * dc_scstride)
pf_x:   .space 8                // PF_ROWS rows of each:
pf_xn:  .space 8
pf_q:   .space 8
pf_k:   .space 8
pf_v:   .space 8
pf_o:   .space 8
pf_ao:  .space 8
pf_g:   .space 8
pf_u:   .space 8
pf_cs:  .space 8                // one RoPE table per row (PF_ROWS * hd)
pf_sc:  .space 8                // PSC_FLOATS
dc_ptrs_end:

// ---------------------------------------------------------------------------
// Optional per-op profile of decode_step (for tests/dec_test.c DEC_ONLY=prof).
// When dec_prof_on != 0, PMARK n adds the TSC ticks since the previous mark to
// dec_prof[n] (so every slot = time of the code BEFORE that mark). Off by
// default: then each mark is one compare + an untaken branch.
// ---------------------------------------------------------------------------
        .data
        .p2align 6
        .globl  dec_prof_on, dec_prof
dec_prof_on: .quad 0
        .globl  dec_fuse
dec_fuse:    .quad 1            // 1 = int8 layers use the fused matvec_q8_multi jobs
                                // (0 = one matvec per matrix; for A/B tests only)
dec_prof_t:  .quad 0            // TSC of the previous mark
        .p2align 6
dec_prof:    .space 8*32        // accumulated ticks per slot (see DP_* below)

        .text

// profile slots
#define DP_START  31            // (only sets the time stamp)
#define DP_EMBED  0
#define DP_ANORM  1
#define DP_WQ     2
#define DP_WK     3
#define DP_WV     4
#define DP_ROPE   5             // QK-norm + RoPE + kv_put
#define DP_ATTN   6
#define DP_WO     7
#define DP_RES1   8
#define DP_FNORM  9
#define DP_WG     10
#define DP_WU     11
#define DP_SILU   12
#define DP_WD     13
#define DP_RES2   14
#define DP_FINAL  15
#define DP_HEAD   16
// prefill slots
#define DP_PEMB   17            // embeddings + RoPE tables of a batch
#define DP_PNORM  18            // the two rmsnorm_rows of a layer
#define DP_PQKV   19            // Q, K, V matmuls + biases
#define DP_PROPE  20            // QK-norm + RoPE of the batch
#define DP_PKV    21            // K/V rows into the cache
#define DP_PATTN  22
#define DP_PWO    23            // Wo matmul + residual
#define DP_PGU    24            // gate + up matmuls
#define DP_PSILU  25
#define DP_PDOWN  26            // down matmul + residual

// PMARK slot: clobbers rax, rcx, rdx (only used between calls, where they are dead)
.macro PMARK slot
        cmp     qword ptr [rip + dec_prof_on], 0
        je      8f
        rdtsc
        shl     rdx, 32
        or      rax, rdx
        mov     rcx, rax
        sub     rax, qword ptr [rip + dec_prof_t]
        mov     qword ptr [rip + dec_prof_t], rcx
.if \slot != DP_START
        add     qword ptr [rip + dec_prof + 8*\slot], rax
.endif
8:
.endm

// ---------------------------------------------------------------------------
// i64 decode_init(Model* m)
// Registers: rbx = m, r12 = running byte offset, rsi = pointer-table cursor.
// ---------------------------------------------------------------------------
// CARVE: dc pointer slot <- current offset; offset += roundup16(rax floats)*4
.macro CARVE
        mov     qword ptr [rsi], r12
        add     rsi, 8
        add     rax, 15
        and     rax, -16                // 64-byte multiples keep every buffer aligned
        shl     rax, 2
        add     r12, rax
.endm

        .globl  decode_init
        .p2align 4
decode_init:
        push    rbx
        push    rsi
        push    rdi
        push    r12
        sub     rsp, 40                 // 8 + 32 + 40 = 80: aligned for calls
        mov     rbx, rcx
        xor     eax, eax
        cmp     rbx, qword ptr [rip + dc_model]
        jne     3f
        lea     rcx, [rip + dc_sig]             // same pointer: same model?
        xor     edx, edx
4:      mov     r8, qword ptr [rbx + rdx*8]
        cmp     r8, qword ptr [rcx + rdx*8]
        jne     3f
        inc     edx
        cmp     edx, 9
        jb      4b
        mov     r8, qword ptr [rbx + M_LAYERS]
        cmp     r8, qword ptr [rcx + 72]
        je      .Ldi_ret                        // yes: nothing to do (eax = 0)
3:
        mov     rcx, qword ptr [rip + dc_block]
        test    rcx, rcx
        jz      1f
        xor     edx, edx                // VirtualFree(block, 0, MEM_RELEASE)
        mov     r8d, MEM_RELEASE
        call    qword ptr [rip + __imp_VirtualFree]
        mov     qword ptr [rip + dc_block], 0
        mov     qword ptr [rip + dc_model], 0
1:      // score stride: roundup8(max_seq) + 8 floats per head
        mov     rax, qword ptr [rbx + M_MAXSEQ]
        add     rax, 15
        and     rax, -8
        mov     qword ptr [rip + dc_scstride], rax
        // carve the block
        xor     r12d, r12d
        lea     rsi, [rip + dc_ptrs]
        mov     rax, qword ptr [rbx + M_DMODEL]                 // x
        CARVE
        mov     rax, qword ptr [rbx + M_DMODEL]                 // xn
        CARVE
        mov     rax, qword ptr [rbx + M_NHEADS]                 // q
        imul    rax, qword ptr [rbx + M_HEADDIM]
        CARVE
        mov     rax, qword ptr [rbx + M_NKV]                    // k
        imul    rax, qword ptr [rbx + M_HEADDIM]
        CARVE
        mov     rax, qword ptr [rbx + M_NHEADS]                 // o
        imul    rax, qword ptr [rbx + M_HEADDIM]
        CARVE
        mov     rax, qword ptr [rbx + M_DMODEL]                 // ao
        CARVE
        mov     rax, qword ptr [rbx + M_DFF]                    // g
        CARVE
        mov     rax, qword ptr [rbx + M_DFF]                    // u
        CARVE
        mov     rax, qword ptr [rbx + M_HEADDIM]                // cs
        CARVE
        mov     rax, qword ptr [rbx + M_HEADDIM]                // inv
        CARVE
        mov     rax, qword ptr [rbx + M_NHEADS]                 // sc
        imul    rax, qword ptr [rip + dc_scstride]
        CARVE
        mov     rax, qword ptr [rbx + M_DMODEL]                 // pf_x
        imul    rax, rax, PF_ROWS
        CARVE
        mov     rax, qword ptr [rbx + M_DMODEL]                 // pf_xn
        imul    rax, rax, PF_ROWS
        CARVE
        mov     rax, qword ptr [rbx + M_NHEADS]                 // pf_q
        imul    rax, qword ptr [rbx + M_HEADDIM]
        imul    rax, rax, PF_ROWS
        CARVE
        mov     rax, qword ptr [rbx + M_NKV]                    // pf_k
        imul    rax, qword ptr [rbx + M_HEADDIM]
        imul    rax, rax, PF_ROWS
        CARVE
        mov     rax, qword ptr [rbx + M_NKV]                    // pf_v
        imul    rax, qword ptr [rbx + M_HEADDIM]
        imul    rax, rax, PF_ROWS
        CARVE
        mov     rax, qword ptr [rbx + M_NHEADS]                 // pf_o
        imul    rax, qword ptr [rbx + M_HEADDIM]
        imul    rax, rax, PF_ROWS
        CARVE
        mov     rax, qword ptr [rbx + M_DMODEL]                 // pf_ao
        imul    rax, rax, PF_ROWS
        CARVE
        mov     rax, qword ptr [rbx + M_DFF]                    // pf_g
        imul    rax, rax, PF_ROWS
        CARVE
        mov     rax, qword ptr [rbx + M_DFF]                    // pf_u
        imul    rax, rax, PF_ROWS
        CARVE
        mov     rax, qword ptr [rbx + M_HEADDIM]                // pf_cs
        imul    rax, rax, PF_ROWS
        CARVE
        mov     eax, PSC_FLOATS                                 // pf_sc
        CARVE
        // one VirtualAlloc for everything (zeroed, committed lazily by the OS)
        xor     ecx, ecx
        mov     rdx, r12
        mov     r8d, MEM_COMMIT_RESERVE
        mov     r9d, PAGE_READWRITE
        call    qword ptr [rip + __imp_VirtualAlloc]
        test    rax, rax
        jz      .Ldi_oom
        mov     qword ptr [rip + dc_block], rax
        lea     rsi, [rip + dc_ptrs]                            // offsets -> pointers
2:      add     qword ptr [rsi], rax
        add     rsi, 8
        lea     rcx, [rip + dc_ptrs_end]
        cmp     rsi, rcx
        jb      2b
        // RoPE inverse frequencies, once per model
        mov     rcx, qword ptr [rip + dc_inv]
        mov     rdx, qword ptr [rbx + M_HEADDIM]
        vmovss  xmm2, dword ptr [rbx + M_ROPEBASE]
        call    rope_inv_freq
        mov     qword ptr [rip + dc_model], rbx
        lea     rcx, [rip + dc_sig]
        xor     edx, edx
5:      mov     r8, qword ptr [rbx + rdx*8]
        mov     qword ptr [rcx + rdx*8], r8
        inc     edx
        cmp     edx, 9
        jb      5b
        mov     r8, qword ptr [rbx + M_LAYERS]
        mov     qword ptr [rcx + 72], r8
        xor     eax, eax
.Ldi_ret:
        add     rsp, 40
        pop     r12
        pop     rdi
        pop     rsi
        pop     rbx
        ret
.Ldi_oom:
        mov     eax, 1
        jmp     .Ldi_ret

// ---------------------------------------------------------------------------
// embed_row (internal leaf): out[0..d) = embedding row of token.
//   rcx = Model*, rdx = token, r8 = out.  Clobbers rax, r9, r10, r11, ymm0.
// Q8: x[j] = (float)q[token*d + j] * scale[j]   (q_llama_decode_step)
// F32: plain copy of row `token` of wte.
// ---------------------------------------------------------------------------
        .p2align 4
embed_row:
        mov     r9, qword ptr [rcx + M_DMODEL]
        mov     rax, qword ptr [rcx + M_WTE + MAT_DATA]
        imul    rdx, r9                                 // token * d
        cmp     qword ptr [rcx + M_WTE + MAT_KIND], MATK_Q8
        je      .Lem_q8
        lea     rax, [rax + rdx*4]
        mov     r10, r9
        and     r10, -8
        xor     r11d, r11d
        test    r10, r10
        jz      2f
1:      vmovups ymm0, ymmword ptr [rax + r11*4]
        vmovups ymmword ptr [r8 + r11*4], ymm0
        add     r11, 8
        cmp     r11, r10
        jb      1b
2:      cmp     r11, r9
        jae     .Lem_ret
        mov     r10d, dword ptr [rax + r11*4]
        mov     dword ptr [r8 + r11*4], r10d
        inc     r11
        jmp     2b
.Lem_q8:
        add     rax, rdx                                // int8 row
        mov     rdx, qword ptr [rcx + M_WTE + MAT_SCALE]
        mov     r10, r9
        and     r10, -8
        xor     r11d, r11d
        test    r10, r10
        jz      4f
3:      vpmovsxbd ymm0, qword ptr [rax + r11]           // 8 int8 -> 8 int32
        vcvtdq2ps ymm0, ymm0                            // exact
        vmulps  ymm0, ymm0, ymmword ptr [rdx + r11*4]   // * scale[j]
        vmovups ymmword ptr [r8 + r11*4], ymm0
        add     r11, 8
        cmp     r11, r10
        jb      3b
4:      cmp     r11, r9
        jae     .Lem_ret
        movsx   r10d, byte ptr [rax + r11]
        vcvtsi2ss xmm0, xmm0, r10d
        vmulss  xmm0, xmm0, dword ptr [rdx + r11*4]
        vmovss  dword ptr [r8 + r11*4], xmm0
        inc     r11
        jmp     4b
.Lem_ret:
        ret

// ---------------------------------------------------------------------------
// Call helpers used by decode_step / prefill. Both keep:
//   rbx = Model*   rsi = KV*   rdi = Layer* of layer r12   r12 = layer index
// and use [rsp+32] / [rsp+40] as the 5th / 6th outgoing argument.
// ---------------------------------------------------------------------------

// RMS x, g, out, n : rmsnorm(x, g, out, n, m->eps). Every operand is a qword
// memory operand written WITHOUT spaces, e.g. [rip+dc_x]: macro arguments are
// space-separated, and the C preprocessor must see the #define names unquoted
// (it never expands macros inside "strings").
.macro RMS x, g, out, n
        mov     rcx, qword ptr \x
        mov     rdx, qword ptr \g
        mov     r8, qword ptr \out
        mov     r9, qword ptr \n
        mov     eax, dword ptr [rbx + M_EPS]
        mov     dword ptr [rsp + 32], eax
        call    rmsnorm
.endm

// MATVEC matoff, x, y : matvec(&layer->mat, x, y)
.macro MATVEC matoff, x, y
        lea     rcx, [rdi + \matoff]
        mov     rdx, qword ptr \x
        mov     r8, qword ptr \y
        call    matvec
.endm

// BIAS boff, y, n : if (layer->bias) vec_add(y, bias, n)
.macro BIAS boff, y, n
        mov     rdx, qword ptr [rdi + \boff]
        test    rdx, rdx
        jz      9f
        mov     rcx, qword ptr \y
        mov     r8, qword ptr \n
        call    vec_add
9:
.endm

// ---------------------------------------------------------------------------
// i64 decode_step(Model* m, KV* kv, i64 token, float* logits)
//
// Frame: 8 pushes + 88 bytes (8 + 64 + 88 = 160, 16-aligned).
//   [rsp+32] 5th outgoing arg   [rsp+40] k slot ptr   [rsp+48] v slot ptr
//   [rsp+56] q_dim  [rsp+64] kv_dim  [rsp+72] d_model  [rsp+80] d_ff
// Registers: rbx = m, rsi = kv, rdi = layer, r12 = l, r13 = pos, r14 = logits,
//            r15 = token, rbp = head_dim
// ---------------------------------------------------------------------------
#define DS_FRAME 88
#define DS_KSLOT 40      // slot index of this position
#define DS_VSLOT 48
#define DS_QD    56
#define DS_KVD   64
#define DS_DM    72
#define DS_DFF   80

        .globl  decode_step
        .p2align 4
decode_step:
        push    rbx
        push    rbp
        push    rsi
        push    rdi
        push    r12
        push    r13
        push    r14
        push    r15
        sub     rsp, DS_FRAME
        mov     rbx, rcx
        mov     rsi, rdx
        mov     r15, r8
        mov     r14, r9
        mov     rcx, rbx
        call    decode_init
        test    eax, eax
        jnz     .Lds_fail
        mov     r13, qword ptr [rsi + KV_LEN]           // pos = kv->len
        cmp     r13, qword ptr [rsi + KV_MAXSEQ]
        jge     .Lds_fail                               // cache full
        PMARK   DP_START
        // dims
        mov     rbp, qword ptr [rbx + M_HEADDIM]
        mov     rax, qword ptr [rbx + M_NHEADS]
        imul    rax, rbp
        mov     qword ptr [rsp + DS_QD], rax
        mov     rax, qword ptr [rbx + M_NKV]
        imul    rax, rbp
        mov     qword ptr [rsp + DS_KVD], rax
        mov     rax, qword ptr [rbx + M_DMODEL]
        mov     qword ptr [rsp + DS_DM], rax
        mov     rax, qword ptr [rbx + M_DFF]
        mov     qword ptr [rsp + DS_DFF], rax
        // x = embedding(token)
        mov     rcx, rbx
        mov     rdx, r15
        mov     r8, qword ptr [rip + dc_x]
        call    embed_row
        // cos/sin table of this position (shared by all layers and heads)
        mov     rcx, qword ptr [rip + dc_cs]
        mov     rdx, qword ptr [rip + dc_inv]
        mov     r8, rbp
        shr     r8, 1
        mov     r9, r13
        call    rope_table
        // attention parameters that do not change per layer
        lea     rax, [rip + dc_args]
        mov     rcx, qword ptr [rip + dc_q]
        mov     qword ptr [rax + AT_Q], rcx
        mov     qword ptr [rax + AT_POS0], r13
        mov     qword ptr [rax + AT_NROWS], 1
        mov     rcx, qword ptr [rbx + M_NHEADS]
        mov     qword ptr [rax + AT_NHEADS], rcx
        mov     rcx, qword ptr [rbx + M_NKV]
        mov     qword ptr [rax + AT_NKV], rcx
        mov     qword ptr [rax + AT_HD], rbp
        mov     rcx, qword ptr [rip + dc_sc]
        mov     qword ptr [rax + AT_SC], rcx
        mov     rcx, qword ptr [rip + dc_scstride]
        mov     qword ptr [rax + AT_SCSTRIDE], rcx
        mov     rcx, qword ptr [rip + dc_o]
        mov     qword ptr [rax + AT_O], rcx

        PMARK   DP_EMBED
        mov     rdi, qword ptr [rbx + M_LAYERS]
        xor     r12d, r12d
.Lds_layer:
        // ---- attention block ----
        RMS     [rip+dc_x], [rdi+L_ATTN_NORM], [rip+dc_xn], [rsp+DS_DM]
        PMARK   DP_ANORM
        // cache slots of this position: slot = pos % cap[l]
        mov     rax, qword ptr [rsi + KV_CAP]
        mov     rcx, qword ptr [rax + r12*8]            // cap
        lea     rax, [rip + dc_args]
        mov     qword ptr [rax + AT_CAP], rcx
        mov     rax, r13
        xor     edx, edx
        div     rcx                                     // rdx = slot
        mov     qword ptr [rsp + DS_KSLOT], rdx         // slot index (for kv_put)
        imul    rdx, qword ptr [rsp + DS_KVD]
        shl     rdx, 2                                  // byte offset of a row-major slot
        mov     rax, qword ptr [rsi + KV_K]
        mov     rax, qword ptr [rax + r12*8]            // K cache of layer l
        lea     rcx, [rip + dc_args]
        mov     qword ptr [rcx + AT_KC], rax
        mov     rax, qword ptr [rsi + KV_V]
        mov     rax, qword ptr [rax + r12*8]
        mov     qword ptr [rcx + AT_VC], rax
        add     rax, rdx
        mov     qword ptr [rsp + DS_VSLOT], rax
        // q, k (into dc_k: stored into the blocked K cache after RoPE) and v
        // (straight into its row-major cache slot), each plus its bias
        cmp     qword ptr [rip + dec_fuse], 0
        je      .Lds_qkv_sep
        cmp     qword ptr [rdi + L_WQ + MAT_KIND], MATK_Q8
        jne     .Lds_qkv_sep
        // int8: ONE pool job for all three (they share xn), bias fused:
        //   matvec_q8_multi({Wq,q,b_q,0}, {Wk,k,b_k,0}, {Wv,vslot,b_v,0}, n=3, xn)
        lea     rcx, [rip + dc_tasks]
        lea     rax, [rdi + L_WQ]
        mov     qword ptr [rcx + 0], rax
        mov     rax, qword ptr [rip + dc_q]
        mov     qword ptr [rcx + 8], rax
        mov     rax, qword ptr [rdi + L_BQ]
        mov     qword ptr [rcx + 16], rax
        mov     qword ptr [rcx + 24], 0
        lea     rax, [rdi + L_WK]
        mov     qword ptr [rcx + 32], rax
        mov     rax, qword ptr [rip + dc_k]
        mov     qword ptr [rcx + 40], rax
        mov     rax, qword ptr [rdi + L_BK]
        mov     qword ptr [rcx + 48], rax
        mov     qword ptr [rcx + 56], 0
        lea     rax, [rdi + L_WV]
        mov     qword ptr [rcx + 64], rax
        mov     rax, qword ptr [rsp + DS_VSLOT]
        mov     qword ptr [rcx + 72], rax
        mov     rax, qword ptr [rdi + L_BV]
        mov     qword ptr [rcx + 80], rax
        mov     qword ptr [rcx + 88], 0
        mov     edx, 3
        mov     r8, qword ptr [rip + dc_xn]
        call    matvec_q8_multi
        PMARK   DP_WQ
        jmp     .Lds_qkv_done
.Lds_qkv_sep:                                           // fp32: one matvec each
        MATVEC  L_WQ, [rip+dc_xn], [rip+dc_q]
        BIAS    L_BQ, [rip+dc_q], [rsp+DS_QD]
        PMARK   DP_WQ
        MATVEC  L_WK, [rip+dc_xn], [rip+dc_k]
        BIAS    L_BK, [rip+dc_k], [rsp+DS_KVD]
        PMARK   DP_WK
        MATVEC  L_WV, [rip+dc_xn], [rsp+DS_VSLOT]
        BIAS    L_BV, [rsp+DS_VSLOT], [rsp+DS_KVD]
        PMARK   DP_WV
.Lds_qkv_done:
        // Qwen3 QK-norm (before RoPE)
        cmp     qword ptr [rbx + M_QKNORM], 0
        je      1f
        mov     rcx, qword ptr [rip + dc_q]
        mov     rdx, qword ptr [rdi + L_QNORM]
        mov     r8, qword ptr [rbx + M_NHEADS]
        mov     r9, rbp
        mov     eax, dword ptr [rbx + M_EPS]
        mov     dword ptr [rsp + 32], eax
        call    rmsnorm_heads
        mov     rcx, qword ptr [rip + dc_k]
        mov     rdx, qword ptr [rdi + L_KNORM]
        mov     r8, qword ptr [rbx + M_NKV]
        mov     r9, rbp
        call    rmsnorm_heads
1:      // RoPE
        mov     rcx, qword ptr [rip + dc_q]
        mov     rdx, qword ptr [rbx + M_NHEADS]
        mov     r8, rbp
        mov     r9, qword ptr [rip + dc_cs]
        call    rope_apply
        mov     rcx, qword ptr [rip + dc_k]
        mov     rdx, qword ptr [rbx + M_NKV]
        mov     r8, rbp
        mov     r9, qword ptr [rip + dc_cs]
        call    rope_apply
        lea     rax, [rip + dc_args]                    // kv_put(kc, cap, slot, k, kv_dim)
        mov     rcx, qword ptr [rax + AT_KC]
        mov     rdx, qword ptr [rax + AT_CAP]
        mov     r8, qword ptr [rsp + DS_KSLOT]
        mov     r9, qword ptr [rip + dc_k]
        mov     rax, qword ptr [rsp + DS_KVD]
        mov     qword ptr [rsp + 32], rax
        call    kv_put
        PMARK   DP_ROPE
        // prefill mode: the last layer's K/V are cached, nothing after matters
        test    r14, r14
        jnz     2f
        lea     rax, [r12 + 1]
        cmp     rax, qword ptr [rbx + M_NLAYERS]
        je      .Lds_done_layers
2:      lea     rcx, [rip + dc_args]
        mov     rax, qword ptr [rdi + L_WINDOW]
        mov     qword ptr [rcx + AT_WINDOW], rax
        call    attention
        PMARK   DP_ATTN
        cmp     qword ptr [rip + dec_fuse], 0
        je      .Lds_ffn_sep
        cmp     qword ptr [rdi + L_WO + MAT_KIND], MATK_Q8
        jne     .Lds_ffn_sep
        // ---- int8: 3 fused jobs for Wo, gate+up, down ----
        // x += o@Wo  (residual fused: y = add = x)
        lea     rcx, [rip + dc_tasks]
        lea     rax, [rdi + L_WO]
        mov     qword ptr [rcx + 0], rax
        mov     rax, qword ptr [rip + dc_x]
        mov     qword ptr [rcx + 8], rax
        mov     qword ptr [rcx + 16], rax
        mov     qword ptr [rcx + 24], 0
        mov     edx, 1
        mov     r8, qword ptr [rip + dc_o]
        call    matvec_q8_multi
        PMARK   DP_WO
        RMS     [rip+dc_x], [rdi+L_FFN_NORM], [rip+dc_xn], [rsp+DS_DM]
        PMARK   DP_FNORM
        // g = SiLU(xn@Wgate) * (xn@Wup)   (a SwiGLU pair: mode 1)
        lea     rcx, [rip + dc_tasks]
        lea     rax, [rdi + L_WGATE]
        mov     qword ptr [rcx + 0], rax
        mov     rax, qword ptr [rip + dc_g]
        mov     qword ptr [rcx + 8], rax
        mov     qword ptr [rcx + 16], 0
        mov     qword ptr [rcx + 24], 1
        lea     rax, [rdi + L_WUP]
        mov     qword ptr [rcx + 32], rax
        mov     qword ptr [rcx + 40], 0
        mov     qword ptr [rcx + 48], 0
        mov     qword ptr [rcx + 56], 0
        mov     edx, 2
        mov     r8, qword ptr [rip + dc_xn]
        call    matvec_q8_multi
        PMARK   DP_WG
        // x += g@Wdown
        lea     rcx, [rip + dc_tasks]
        lea     rax, [rdi + L_WDOWN]
        mov     qword ptr [rcx + 0], rax
        mov     rax, qword ptr [rip + dc_x]
        mov     qword ptr [rcx + 8], rax
        mov     qword ptr [rcx + 16], rax
        mov     qword ptr [rcx + 24], 0
        mov     edx, 1
        mov     r8, qword ptr [rip + dc_g]
        call    matvec_q8_multi
        PMARK   DP_WD
        jmp     .Lds_layer_next
.Lds_ffn_sep:                                           // fp32: separate ops
        MATVEC  L_WO, [rip+dc_o], [rip+dc_ao]
        PMARK   DP_WO
        mov     rcx, qword ptr [rip + dc_x]                     // residual
        mov     rdx, qword ptr [rip + dc_ao]
        mov     r8, qword ptr [rsp + DS_DM]
        call    vec_add
        PMARK   DP_RES1
        // ---- feed-forward block (SwiGLU) ----
        RMS     [rip+dc_x], [rdi+L_FFN_NORM], [rip+dc_xn], [rsp+DS_DM]
        PMARK   DP_FNORM
        MATVEC  L_WGATE, [rip+dc_xn], [rip+dc_g]
        PMARK   DP_WG
        MATVEC  L_WUP, [rip+dc_xn], [rip+dc_u]
        PMARK   DP_WU
        mov     rcx, qword ptr [rip + dc_g]
        mov     rdx, qword ptr [rip + dc_u]
        mov     r8, qword ptr [rsp + DS_DFF]
        call    silu_mul
        PMARK   DP_SILU
        MATVEC  L_WDOWN, [rip+dc_g], [rip+dc_ao]
        PMARK   DP_WD
        mov     rcx, qword ptr [rip + dc_x]
        mov     rdx, qword ptr [rip + dc_ao]
        mov     r8, qword ptr [rsp + DS_DM]
        call    vec_add
        PMARK   DP_RES2
.Lds_layer_next:
        add     rdi, L_SIZE
        inc     r12
        cmp     r12, qword ptr [rbx + M_NLAYERS]
        jb      .Lds_layer
.Lds_done_layers:
        test    r14, r14
        jz      3f
        RMS     [rip+dc_x], [rbx+M_FINALNORM], [rip+dc_xn], [rsp+DS_DM]
        PMARK   DP_FINAL
        lea     rcx, [rbx + M_LMHEAD]
        mov     rdx, qword ptr [rip + dc_xn]
        mov     r8, r14
        call    matvec
        PMARK   DP_HEAD
3:      lea     rax, [r13 + 1]
        mov     qword ptr [rsi + KV_LEN], rax           // kv->len = pos + 1
        xor     eax, eax
.Lds_ret:
        add     rsp, DS_FRAME
        pop     r15
        pop     r14
        pop     r13
        pop     r12
        pop     rdi
        pop     rsi
        pop     rbp
        pop     rbx
        ret
.Lds_fail:
        mov     eax, 1
        jmp     .Lds_ret

// ---------------------------------------------------------------------------
// put_k_rows (internal): K rows src[i] (kv_dim floats each) -> cache slots
// (pos + i) % cap, via kv_put (blocked K layout, see ops.S).
//   rcx = K cache, rdx = cap, r8 = first position, r9 = src,
//   [rsp+40] = kv_dim, [rsp+48] = n
// ---------------------------------------------------------------------------
        .p2align 4
put_k_rows:
        push    rbx
        push    rsi
        push    rdi
        push    r12
        push    r13
        push    r14
        sub     rsp, 40                         // 8 + 48 + 40 = 96
        mov     rbx, rcx
        mov     rsi, rdx
        mov     rdi, r8
        mov     r12, r9
        mov     r13, qword ptr [rsp + 40 + 48 + 40]     // kv_dim
        mov     r14, qword ptr [rsp + 40 + 48 + 48]     // n
1:      test    r14, r14
        jz      2f
        mov     rax, rdi
        xor     edx, edx
        div     rsi                             // slot
        mov     r8, rdx
        mov     rcx, rbx
        mov     rdx, rsi
        mov     r9, r12
        mov     qword ptr [rsp + 32], r13
        call    kv_put
        lea     r12, [r12 + r13*4]
        inc     rdi
        dec     r14
        jmp     1b
2:      add     rsp, 40
        pop     r14
        pop     r13
        pop     r12
        pop     rdi
        pop     rsi
        pop     rbx
        ret

// ---------------------------------------------------------------------------
// Row-parallel versions of the per-row ops used by prefill. Every row is still
// computed by the same single-row code (rms_core / SILU8), only different rows
// run on different threads, so the results are bit-identical.
//   void par_rmsnorm_rows(x, g, out, i64 n, float eps, i64 rows)  (= rmsnorm_rows)
//   void par_silu_rows(float* g, const float* u, i64 n, i64 rows)  (= silu_mul of n*rows)
// pj_ctx: +0 kind (0 rms, 1 silu), +8 a (x / g), +16 b (gain / u), +24 out,
//         +32 n, +40 eps (f32)
// ---------------------------------------------------------------------------
        .bss
        .p2align 6
pj_ctx: .space 64
        .text
        .p2align 4
pj_chunk:                                       // pool fn (ctx, r0, r1)
        sub     rsp, 56
        mov     r10, rcx
        mov     r11, r8
        sub     r11, rdx                        // rows of this chunk
        mov     r9, qword ptr [r10 + 32]        // n
        mov     rax, rdx
        imul    rax, r9
        shl     rax, 2                          // byte offset of row r0
        cmp     qword ptr [r10], 0
        jne     1f
        mov     rcx, qword ptr [r10 + 8]
        add     rcx, rax                        // x + r0*n
        mov     rdx, qword ptr [r10 + 16]       // gain (shared by all rows)
        mov     r8, qword ptr [r10 + 24]
        add     r8, rax                         // out + r0*n
        mov     eax, dword ptr [r10 + 40]
        mov     dword ptr [rsp + 32], eax       // eps
        mov     qword ptr [rsp + 40], r11       // rows
        call    rmsnorm_rows
        jmp     2f
1:      mov     rcx, qword ptr [r10 + 8]
        add     rcx, rax
        mov     rdx, qword ptr [r10 + 16]
        add     rdx, rax
        mov     r8, r11
        imul    r8, r9                          // floats
        call    silu_mul
2:      add     rsp, 56
        ret

// pj_run (internal): pool_run(pj_chunk, pj_ctx, rows, grain = ceil(rows/16))
//   rcx = rows
pj_run:
        sub     rsp, 40
        mov     r8, rcx
        lea     r9, [rcx + 15]
        shr     r9, 4
        lea     rcx, [rip + pj_chunk]
        lea     rdx, [rip + pj_ctx]
        call    pool_run
        add     rsp, 40
        ret

        .globl  par_rmsnorm_rows
par_rmsnorm_rows:
        sub     rsp, 40
        lea     rax, [rip + pj_ctx]
        mov     qword ptr [rax], 0
        mov     qword ptr [rax + 8], rcx
        mov     qword ptr [rax + 16], rdx
        mov     qword ptr [rax + 24], r8
        mov     qword ptr [rax + 32], r9
        mov     r10d, dword ptr [rsp + 40 + 40]
        mov     dword ptr [rax + 40], r10d
        mov     rcx, qword ptr [rsp + 40 + 48]  // rows
        call    pj_run
        add     rsp, 40
        ret

        .globl  par_silu_rows
par_silu_rows:
        sub     rsp, 40
        lea     rax, [rip + pj_ctx]
        mov     qword ptr [rax], 1
        mov     qword ptr [rax + 8], rcx
        mov     qword ptr [rax + 16], rdx
        mov     qword ptr [rax + 32], r8
        mov     rcx, r9
        call    pj_run
        add     rsp, 40
        ret

// ---------------------------------------------------------------------------
// i64 prefill(Model* m, KV* kv, const i32* ids, i64 n)
//
// Per batch of M <= PF_ROWS tokens at positions pos0..pos0+M-1, every layer:
//   XN = rmsnorm rows ; Q,K,V = matmul_rows + bias ; QK-norm ; RoPE per row
//   K/V rows -> cache ; attention of every row (causal: row i sees keys <= pos0+i)
//   X += O@Wo ; XN = rmsnorm rows ; X += (SiLU(XN@Wg) * XN@Wu) @ Wd
// Attention of the whole batch at once needs every row's K/V in the cache first.
// That is safe unless the batch would overwrite ring slots an earlier row of the
// batch still reads (a sliding-window layer whose ring wraps inside the batch):
// then rows are appended and attended one at a time, in order (bit-identical,
// just less parallel).
//
// Frame: 8 pushes + 120 (8 + 64 + 120 = 192).
//   [rsp+32],[rsp+40] outgoing args 5/6
//   [rsp+48] M  [rsp+56] q_dim  [rsp+64] kv_dim  [rsp+72] d_model  [rsp+80] d_ff
//   [rsp+88] row loop counter   [rsp+96] score stride   [rsp+104] rows per attention call
//   [rsp+112] last-layer flag
// Registers: rbx = m, rsi = kv, rdi = layer, r12 = l, r13 = pos0, r14 = cap of layer,
//            r15 = ids, rbp = tokens left
// ---------------------------------------------------------------------------
#define PF_FRAME 120
#define PF_M     48
#define PF_QD    56
#define PF_KVD   64
#define PF_DM    72
#define PF_DFF   80
#define PF_I     88
#define PF_SCS   96
#define PF_RPC   104
#define PF_LAST  112

// MATROWS matoff, X, Y : matmul_rows(&layer->mat, X, M, Y)
.macro MATROWS matoff, X, Y
        lea     rcx, [rdi + \matoff]
        mov     rdx, qword ptr \X
        mov     r8, qword ptr [rsp + PF_M]
        mov     r9, qword ptr \Y
        call    matmul_rows
.endm

// BIASROWS boff, Y, n : if (layer->bias) add_rows(Y, bias, n, M)
.macro BIASROWS boff, Y, n
        mov     rdx, qword ptr [rdi + \boff]
        test    rdx, rdx
        jz      9f
        mov     rcx, qword ptr \Y
        mov     r8, qword ptr \n
        mov     r9, qword ptr [rsp + PF_M]
        call    add_rows
9:
.endm

        .globl  prefill
        .p2align 4
prefill:
        push    rbx
        push    rbp
        push    rsi
        push    rdi
        push    r12
        push    r13
        push    r14
        push    r15
        sub     rsp, PF_FRAME
        mov     rbx, rcx
        mov     rsi, rdx
        mov     r15, r8
        mov     rbp, r9
        mov     rcx, rbx
        call    decode_init
        test    eax, eax
        jnz     .Lpf_fail
        mov     rcx, qword ptr [rbx + M_HEADDIM]
        mov     rax, qword ptr [rbx + M_NHEADS]
        imul    rax, rcx
        mov     qword ptr [rsp + PF_QD], rax
        mov     rax, qword ptr [rbx + M_NKV]
        imul    rax, rcx
        mov     qword ptr [rsp + PF_KVD], rax
        mov     rax, qword ptr [rbx + M_DMODEL]
        mov     qword ptr [rsp + PF_DM], rax
        mov     rax, qword ptr [rbx + M_DFF]
        mov     qword ptr [rsp + PF_DFF], rax

.Lpf_batch:
        PMARK   DP_START
        test    rbp, rbp
        jle     .Lpf_ok
        mov     r13, qword ptr [rsi + KV_LEN]           // pos0
        mov     rax, qword ptr [rsi + KV_MAXSEQ]
        sub     rax, r13                                // room left in the cache
        jle     .Lpf_fail
        mov     rcx, rbp
        cmp     rcx, PF_ROWS
        jbe     1f
        mov     ecx, PF_ROWS
1:      cmp     rcx, rax
        jbe     2f
        mov     rcx, rax
2:      mov     qword ptr [rsp + PF_M], rcx             // M rows in this batch
        // score scratch geometry: stride = roundup8(pos0+M+7) + 8,
        // rows per attention call = max(1, PSC_FLOATS / (n_heads*stride))
        lea     rax, [r13 + rcx + 23]
        and     rax, -8
        mov     qword ptr [rsp + PF_SCS], rax
        imul    rax, qword ptr [rbx + M_NHEADS]
        mov     rcx, rax
        mov     eax, PSC_FLOATS
        xor     edx, edx
        div     rcx
        test    rax, rax
        jnz     3f
        mov     eax, 1
3:      mov     qword ptr [rsp + PF_RPC], rax

        // embeddings + RoPE tables of the batch rows
        mov     qword ptr [rsp + PF_I], 0
.Lpf_embed:
        mov     rax, qword ptr [rsp + PF_I]
        movsxd  rdx, dword ptr [r15 + rax*4]            // token id (i32)
        mov     r8, rax
        imul    r8, qword ptr [rsp + PF_DM]
        shl     r8, 2
        add     r8, qword ptr [rip + pf_x]
        mov     rcx, rbx
        call    embed_row
        mov     rax, qword ptr [rsp + PF_I]
        mov     r8, qword ptr [rbx + M_HEADDIM]
        mov     rcx, rax
        imul    rcx, r8
        shl     rcx, 2
        add     rcx, qword ptr [rip + pf_cs]            // table of row i
        mov     rdx, qword ptr [rip + dc_inv]
        shr     r8, 1
        lea     r9, [r13 + rax]                         // position pos0 + i
        call    rope_table
        mov     rax, qword ptr [rsp + PF_I]
        inc     rax
        mov     qword ptr [rsp + PF_I], rax
        cmp     rax, qword ptr [rsp + PF_M]
        jb      .Lpf_embed

        PMARK   DP_PEMB
        mov     rdi, qword ptr [rbx + M_LAYERS]
        xor     r12d, r12d
.Lpf_layer:
        lea     rax, [r12 + 1]
        xor     ecx, ecx
        cmp     rax, qword ptr [rbx + M_NLAYERS]
        sete    cl
        mov     qword ptr [rsp + PF_LAST], rcx
        // XN = rmsnorm(X) row by row
        mov     rcx, qword ptr [rip + pf_x]
        mov     rdx, qword ptr [rdi + L_ATTN_NORM]
        mov     r8, qword ptr [rip + pf_xn]
        mov     r9, qword ptr [rsp + PF_DM]
        mov     eax, dword ptr [rbx + M_EPS]
        mov     dword ptr [rsp + 32], eax
        mov     rax, qword ptr [rsp + PF_M]
        mov     qword ptr [rsp + 40], rax
        call    par_rmsnorm_rows
        PMARK   DP_PNORM
        // Q, K, V for the whole batch
        MATROWS  L_WQ, [rip+pf_xn], [rip+pf_q]
        BIASROWS L_BQ, [rip+pf_q], [rsp+PF_QD]
        MATROWS  L_WK, [rip+pf_xn], [rip+pf_k]
        BIASROWS L_BK, [rip+pf_k], [rsp+PF_KVD]
        MATROWS  L_WV, [rip+pf_xn], [rip+pf_v]
        BIASROWS L_BV, [rip+pf_v], [rsp+PF_KVD]
        PMARK   DP_PQKV
        cmp     qword ptr [rbx + M_QKNORM], 0
        je      1f
        mov     rcx, qword ptr [rip + pf_q]             // all rows' heads at once
        mov     rdx, qword ptr [rdi + L_QNORM]
        mov     r8, qword ptr [rbx + M_NHEADS]
        imul    r8, qword ptr [rsp + PF_M]
        mov     r9, qword ptr [rbx + M_HEADDIM]
        mov     eax, dword ptr [rbx + M_EPS]
        mov     dword ptr [rsp + 32], eax
        call    rmsnorm_heads
        mov     rcx, qword ptr [rip + pf_k]
        mov     rdx, qword ptr [rdi + L_KNORM]
        mov     r8, qword ptr [rbx + M_NKV]
        imul    r8, qword ptr [rsp + PF_M]
        mov     r9, qword ptr [rbx + M_HEADDIM]
        call    rmsnorm_heads
1:      mov     rcx, qword ptr [rip + pf_q]             // RoPE, row i at pos0+i
        mov     rdx, qword ptr [rbx + M_NHEADS]
        mov     r8, qword ptr [rbx + M_HEADDIM]
        mov     r9, qword ptr [rip + pf_cs]
        mov     rax, qword ptr [rsp + PF_M]
        mov     qword ptr [rsp + 32], rax
        call    rope_apply_rows
        mov     rcx, qword ptr [rip + pf_k]
        mov     rdx, qword ptr [rbx + M_NKV]
        mov     r8, qword ptr [rbx + M_HEADDIM]
        mov     r9, qword ptr [rip + pf_cs]
        mov     rax, qword ptr [rsp + PF_M]
        mov     qword ptr [rsp + 32], rax
        call    rope_apply_rows
        PMARK   DP_PROPE

        // this layer's cache
        mov     rax, qword ptr [rsi + KV_CAP]
        mov     r14, qword ptr [rax + r12*8]            // cap
        lea     rax, [rip + dc_args]
        mov     qword ptr [rax + AT_CAP], r14
        mov     rcx, qword ptr [rsi + KV_K]
        mov     rcx, qword ptr [rcx + r12*8]
        mov     qword ptr [rax + AT_KC], rcx
        mov     rcx, qword ptr [rsi + KV_V]
        mov     rcx, qword ptr [rcx + r12*8]
        mov     qword ptr [rax + AT_VC], rcx
        mov     rcx, qword ptr [rdi + L_WINDOW]
        mov     qword ptr [rax + AT_WINDOW], rcx
        mov     rcx, qword ptr [rbx + M_NHEADS]
        mov     qword ptr [rax + AT_NHEADS], rcx
        mov     rcx, qword ptr [rbx + M_NKV]
        mov     qword ptr [rax + AT_NKV], rcx
        mov     rcx, qword ptr [rbx + M_HEADDIM]
        mov     qword ptr [rax + AT_HD], rcx

        mov     rax, r13
        add     rax, qword ptr [rsp + PF_M]
        cmp     rax, r14
        ja      .Lpf_rowwise                            // ring wraps inside the batch

        // ---- batch path: all K/V rows into slots pos0.. (no wrap) ----
        lea     rax, [rip + dc_args]
        mov     rcx, qword ptr [rax + AT_KC]
        mov     rdx, r14
        mov     r8, r13
        mov     r9, qword ptr [rip + pf_k]
        mov     rax, qword ptr [rsp + PF_KVD]
        mov     qword ptr [rsp + 32], rax
        mov     rax, qword ptr [rsp + PF_M]
        mov     qword ptr [rsp + 40], rax
        call    put_k_rows
        mov     rcx, r13
        imul    rcx, qword ptr [rsp + PF_KVD]
        shl     rcx, 2
        lea     rax, [rip + dc_args]
        add     rcx, qword ptr [rax + AT_VC]
        mov     rdx, qword ptr [rip + pf_v]
        mov     r8, qword ptr [rsp + PF_KVD]
        imul    r8, qword ptr [rsp + PF_M]
        call    vec_copy
        PMARK   DP_PKV
        cmp     qword ptr [rsp + PF_LAST], 0
        jne     .Lpf_batch_done
        // attention in chunks of PF_RPC rows (bounded score scratch)
        lea     rax, [rip + dc_args]
        mov     rcx, qword ptr [rip + pf_sc]
        mov     qword ptr [rax + AT_SC], rcx
        mov     rcx, qword ptr [rsp + PF_SCS]
        mov     qword ptr [rax + AT_SCSTRIDE], rcx
        mov     qword ptr [rsp + PF_I], 0
.Lpf_attn_chunk:
        mov     rcx, qword ptr [rsp + PF_I]             // first row r0
        mov     rdx, qword ptr [rsp + PF_M]
        sub     rdx, rcx                                // rows left
        jle     .Lpf_ffn
        cmp     rdx, qword ptr [rsp + PF_RPC]
        jbe     4f
        mov     rdx, qword ptr [rsp + PF_RPC]
4:      lea     rax, [rip + dc_args]
        mov     qword ptr [rax + AT_NROWS], rdx
        lea     r8, [r13 + rcx]
        mov     qword ptr [rax + AT_POS0], r8
        mov     r8, rcx
        imul    r8, qword ptr [rsp + PF_QD]
        shl     r8, 2
        mov     r9, qword ptr [rip + pf_q]
        add     r9, r8
        mov     qword ptr [rax + AT_Q], r9
        mov     r9, qword ptr [rip + pf_o]
        add     r9, r8
        mov     qword ptr [rax + AT_O], r9
        add     rcx, rdx
        mov     qword ptr [rsp + PF_I], rcx
        lea     rcx, [rip + dc_args]
        call    attention
        jmp     .Lpf_attn_chunk

        // ---- row-wise path (ring wraps): append row i, attend row i ----
.Lpf_rowwise:
        lea     rax, [rip + dc_args]
        mov     rcx, qword ptr [rip + dc_sc]
        mov     qword ptr [rax + AT_SC], rcx
        mov     rcx, qword ptr [rip + dc_scstride]
        mov     qword ptr [rax + AT_SCSTRIDE], rcx
        mov     qword ptr [rax + AT_NROWS], 1
        mov     qword ptr [rsp + PF_I], 0
.Lpf_row:
        mov     rax, qword ptr [rsp + PF_I]
        lea     rax, [r13 + rax]                        // position
        xor     edx, edx
        div     r14                                     // rdx = slot
        imul    rdx, qword ptr [rsp + PF_KVD]
        shl     rdx, 2
        mov     qword ptr [rsp + 40], rdx               // (scratch) slot byte offset
        mov     r8, qword ptr [rsp + PF_I]
        imul    r8, qword ptr [rsp + PF_KVD]
        shl     r8, 2                                   // row byte offset in pf_k/pf_v
        mov     qword ptr [rsp + 32], r8
        lea     rax, [rip + dc_args]                    // K row i -> slot of pos0+i
        mov     rcx, qword ptr [rax + AT_KC]
        mov     rdx, r14
        mov     r8, qword ptr [rsp + PF_I]
        add     r8, r13
        mov     r9, qword ptr [rip + pf_k]
        add     r9, qword ptr [rsp + 32]
        mov     rax, qword ptr [rsp + PF_KVD]
        mov     qword ptr [rsp + 32], rax
        mov     qword ptr [rsp + 40], 1
        call    put_k_rows
        mov     r8, qword ptr [rsp + PF_I]              // (recompute the row offset)
        imul    r8, qword ptr [rsp + PF_KVD]
        shl     r8, 2
        mov     qword ptr [rsp + 32], r8
        mov     rax, qword ptr [rsp + PF_I]
        add     rax, r13
        xor     edx, edx
        div     r14
        imul    rdx, qword ptr [rsp + PF_KVD]
        shl     rdx, 2
        mov     qword ptr [rsp + 40], rdx               // V slot byte offset
        lea     rax, [rip + dc_args]
        mov     rcx, qword ptr [rax + AT_VC]
        add     rcx, qword ptr [rsp + 40]
        mov     rdx, qword ptr [rip + pf_v]
        add     rdx, qword ptr [rsp + 32]
        mov     r8, qword ptr [rsp + PF_KVD]
        call    vec_copy
        cmp     qword ptr [rsp + PF_LAST], 0
        jne     5f
        mov     rcx, qword ptr [rsp + PF_I]
        lea     rax, [rip + dc_args]
        lea     r8, [r13 + rcx]
        mov     qword ptr [rax + AT_POS0], r8
        imul    rcx, qword ptr [rsp + PF_QD]
        shl     rcx, 2
        mov     r9, qword ptr [rip + pf_q]
        add     r9, rcx
        mov     qword ptr [rax + AT_Q], r9
        mov     r9, qword ptr [rip + pf_o]
        add     r9, rcx
        mov     qword ptr [rax + AT_O], r9
        lea     rcx, [rip + dc_args]
        call    attention
5:      mov     rax, qword ptr [rsp + PF_I]
        inc     rax
        mov     qword ptr [rsp + PF_I], rax
        cmp     rax, qword ptr [rsp + PF_M]
        jb      .Lpf_row
        cmp     qword ptr [rsp + PF_LAST], 0
        jne     .Lpf_batch_done

        // ---- output projection + FFN for the whole batch ----
.Lpf_ffn:
        PMARK   DP_PATTN
        MATROWS L_WO, [rip+pf_o], [rip+pf_ao]
        mov     rcx, qword ptr [rip + pf_x]
        mov     rdx, qword ptr [rip + pf_ao]
        mov     r8, qword ptr [rsp + PF_DM]
        imul    r8, qword ptr [rsp + PF_M]
        call    vec_add
        PMARK   DP_PWO
        mov     rcx, qword ptr [rip + pf_x]
        mov     rdx, qword ptr [rdi + L_FFN_NORM]
        mov     r8, qword ptr [rip + pf_xn]
        mov     r9, qword ptr [rsp + PF_DM]
        mov     eax, dword ptr [rbx + M_EPS]
        mov     dword ptr [rsp + 32], eax
        mov     rax, qword ptr [rsp + PF_M]
        mov     qword ptr [rsp + 40], rax
        call    par_rmsnorm_rows
        PMARK   DP_PNORM
        MATROWS L_WGATE, [rip+pf_xn], [rip+pf_g]
        MATROWS L_WUP, [rip+pf_xn], [rip+pf_u]
        PMARK   DP_PGU
        mov     rcx, qword ptr [rip + pf_g]
        mov     rdx, qword ptr [rip + pf_u]
        mov     r8, qword ptr [rsp + PF_DFF]
        mov     r9, qword ptr [rsp + PF_M]
        call    par_silu_rows
        PMARK   DP_PSILU
        MATROWS L_WDOWN, [rip+pf_g], [rip+pf_ao]
        mov     rcx, qword ptr [rip + pf_x]
        mov     rdx, qword ptr [rip + pf_ao]
        mov     r8, qword ptr [rsp + PF_DM]
        imul    r8, qword ptr [rsp + PF_M]
        call    vec_add
        PMARK   DP_PDOWN
        add     rdi, L_SIZE
        inc     r12
        cmp     r12, qword ptr [rbx + M_NLAYERS]
        jb      .Lpf_layer

.Lpf_batch_done:
        mov     rax, qword ptr [rsp + PF_M]
        lea     rcx, [r13 + rax]
        mov     qword ptr [rsi + KV_LEN], rcx           // kv->len = pos0 + M
        lea     r15, [r15 + rax*4]                      // ids += M
        sub     rbp, rax
        jmp     .Lpf_batch
.Lpf_ok:
        xor     eax, eax
.Lpf_ret:
        add     rsp, PF_FRAME
        pop     r15
        pop     r14
        pop     r13
        pop     r12
        pop     rdi
        pop     rsi
        pop     rbp
        pop     rbx
        ret
.Lpf_fail:
        mov     eax, 1
        jmp     .Lpf_ret
