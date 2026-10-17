// sample.S - token sampling, an exact port of ../sampler.c (owner: decode agent)
//
// Exported (Win64 ABI):
//   void  rng_seed(Rng* r, u64 seed)        r->state = seed ? seed : 0x9E3779B97F4A7C15
//                                           (= rng_create; gen.c seeds with seed + sample#)
//   float rng_uniform(Rng* r)               xorshift64, top 53 bits -> double -> float
//   i64   sample_argmax(const float* logits, i64 V)   first index of the maximum
//   i64   sample_token(const float* logits, i64 V, float temperature,
//                      i64 top_k, float top_p, Rng* rng)
//
// sample_token returns exactly the token sampler.c returns for the same logits
// and RNG state, and advances the RNG the same way (one draw unless greedy):
//   temperature <= 0  -> argmax, no RNG draw
//   e_j = expf((l_j - max) / T)      (expf bit-exact with the C library's, see ops.S)
//   sum = e_0 + e_1 + ...            (float, strictly in index order, like C)
//   p_j = e_j / sum
//   order p descending, ties by ascending id (a total order)
//   keep = top-k / top-p cut on that order (keep_count in sampler.c)
//   z = sum of the kept p (float, in order) ; u = rng_uniform * z
//   first i with u <= p_0+..+p_i (float running sum)  -> its id, else the last kept
// Like the C fast path, only the head of the order is materialized: a histogram
// of the float exponents picks a threshold 2^e that provably covers the cut, the
// candidates p >= 2^e are sorted, and keep_count verifies the cut lies inside
// them (else everything is sorted). Every float sum then runs over the same
// entries in the same order as a full sort, so the token is bit-for-bit the same.
//
// Sorting: each candidate becomes one u64 key  (~bits(p) << 32) | id : ascending
// keys = descending p (p >= 0, so its bit pattern is monotone in its value), and
// equal p fall back to ascending id. Small sets use insertion sort, larger ones
// an LSD radix sort (8-bit digits, digits shared by all keys are skipped).

#include "model.inc"
        .intel_syntax noprefix

#define MEM_COMMIT_RESERVE 0x3000
#define MEM_RELEASE        0x8000
#define PAGE_READWRITE     0x04
#define INSERTION_MAX      32

        .section .rdata,"dr"
        .p2align 3
S_ONE:    .float 1.0
S_HALF53: .double 1.1102230246251565e-16        // 1.0 / 9007199254740992.0 = 2^-53
S_MARGIN: .double 1e-3

        .bss
        .p2align 6
sp_cap:   .space 8              // capacity (entries) of the scratch below
sp_block: .space 8              // VirtualAlloc base
sp_p:     .space 8              // float p[V]
sp_keys:  .space 8              // u64 keys[V]
sp_tmp:   .space 8              // u64 tmp[V] (radix sort ping-pong)
sp_cnt:   .space 256*4*4        // exponent histogram: 4 interleaved copies of 256 counts
sp_mass:  .space 256*8*4        //                    and of 256 double masses
sp_hist:  .space 8*256*4        // radix: 8 digit histograms of 256 u32

        .text

// ---------------------------------------------------------------------------
// void rng_seed(Rng* r, u64 seed)
// ---------------------------------------------------------------------------
        .globl  rng_seed
        .p2align 4
rng_seed:
        test    rdx, rdx
        jnz     1f
        mov     rdx, 0x9E3779B97F4A7C15         // avoid the all-zero fixed point
1:      mov     qword ptr [rcx], rdx
        ret

// ---------------------------------------------------------------------------
// float rng_uniform(Rng* r)
//   x ^= x << 13; x ^= x >> 7; x ^= x << 17;  return (float)((x >> 11) * 2^-53)
// ---------------------------------------------------------------------------
        .globl  rng_uniform
        .p2align 4
rng_uniform:
        mov     rax, qword ptr [rcx]
        mov     rdx, rax
        shl     rdx, 13
        xor     rax, rdx
        mov     rdx, rax
        shr     rdx, 7
        xor     rax, rdx
        mov     rdx, rax
        shl     rdx, 17
        xor     rax, rdx
        mov     qword ptr [rcx], rax
        shr     rax, 11                         // < 2^53: exact as a signed int64 -> double
        vcvtsi2sd xmm0, xmm0, rax
        vmulsd  xmm0, xmm0, qword ptr [rip + S_HALF53]  // exact (power of two)
        vcvtsd2ss xmm0, xmm0, xmm0              // round to float (may give 1.0f, like C)
        ret

// ---------------------------------------------------------------------------
// i64 sample_argmax(const float* l, i64 V)
// C: best = 0; for j: if (l[j] > l[best]) best = j  -> the FIRST index holding the
// maximum. Pass 1: vector max. Pass 2: first index equal to it.
// ---------------------------------------------------------------------------
        .globl  sample_argmax
        .p2align 4
sample_argmax:
        xor     eax, eax
        test    rdx, rdx
        jle     .Lam_ret
        vbroadcastss ymm0, dword ptr [rcx]
        mov     r8, rdx
        and     r8, -8
        xor     r9d, r9d
        test    r8, r8
        jz      2f
1:      vmaxps  ymm0, ymm0, ymmword ptr [rcx + r9*4]
        add     r9, 8
        cmp     r9, r8
        jb      1b
2:      vextractf128 xmm1, ymm0, 1
        vmaxps  xmm0, xmm0, xmm1
        vpermilps xmm1, xmm0, 0x4E
        vmaxps  xmm0, xmm0, xmm1
        vpermilps xmm1, xmm0, 0xB1
        vmaxps  xmm0, xmm0, xmm1
3:      cmp     r9, rdx
        jae     4f
        vmaxss  xmm0, xmm0, dword ptr [rcx + r9*4]
        inc     r9
        jmp     3b
4:      vbroadcastss ymm0, xmm0                 // the maximum
        xor     r9d, r9d
        test    r8, r8
        jz      6f
5:      vcmpeqps ymm1, ymm0, ymmword ptr [rcx + r9*4]
        vmovmskps eax, ymm1
        test    eax, eax
        jnz     7f
        add     r9, 8
        cmp     r9, r8
        jb      5b
6:      cmp     r9, rdx
        jae     8f
        vucomiss xmm0, dword ptr [rcx + r9*4]
        je      9f
        inc     r9
        jmp     6b
7:      tzcnt   eax, eax                        // first matching lane
        add     rax, r9
        jmp     .Lam_ret
8:      xor     eax, eax                        // (NaNs only) fall back to 0
        jmp     .Lam_ret
9:      mov     rax, r9
.Lam_ret:
        vzeroupper
        ret

// ---------------------------------------------------------------------------
// sp_reserve (internal): make the scratch hold V entries.  rcx = V.
// Returns rax = 0 ok / 1 out of memory. Standard Win64 function.
// ---------------------------------------------------------------------------
        .p2align 4
sp_reserve:
        push    rbx
        sub     rsp, 32
        xor     eax, eax
        cmp     rcx, qword ptr [rip + sp_cap]
        jbe     2f
        mov     rbx, rcx
        mov     rcx, qword ptr [rip + sp_block]
        test    rcx, rcx
        jz      1f
        xor     edx, edx
        mov     r8d, MEM_RELEASE
        call    qword ptr [rip + __imp_VirtualFree]
        mov     qword ptr [rip + sp_cap], 0
1:      lea     rdx, [rbx + 16]                 // p (4 B) + keys (8) + tmp (8) per entry
        imul    rdx, rdx, 20
        add     rdx, 256
        xor     ecx, ecx
        mov     r8d, MEM_COMMIT_RESERVE
        mov     r9d, PAGE_READWRITE
        call    qword ptr [rip + __imp_VirtualAlloc]
        test    rax, rax
        jz      3f
        mov     qword ptr [rip + sp_block], rax
        lea     rcx, [rbx + 16]
        mov     qword ptr [rip + sp_keys], rax          // keys first: 8-aligned
        lea     rax, [rax + rcx*8]
        mov     qword ptr [rip + sp_tmp], rax
        lea     rax, [rax + rcx*8]
        mov     qword ptr [rip + sp_p], rax
        mov     qword ptr [rip + sp_cap], rbx
        xor     eax, eax
2:      add     rsp, 32
        pop     rbx
        ret
3:      mov     eax, 1
        jmp     2b

// ---------------------------------------------------------------------------
// sort_keys (internal leaf): sort n u64 keys ascending in place.
//   rcx = keys, rdx = tmp (n entries), r8 = n.
//   Clobbers rax, rcx, rdx, r8-r11 and saves/uses rbx, rsi, rdi, r12-r14.
// ---------------------------------------------------------------------------
        .p2align 4
sort_keys:
        push    rbx
        push    rsi
        push    rdi
        push    r12
        push    r13
        push    r14
        cmp     r8, 1
        jbe     .Lsk_ret
        cmp     r8, INSERTION_MAX
        ja      .Lsk_radix
        // insertion sort
        mov     r9d, 1                          // i
1:      mov     rax, qword ptr [rcx + r9*8]     // key to insert
        mov     r10, r9
2:      test    r10, r10
        jz      3f
        mov     r11, qword ptr [rcx + r10*8 - 8]
        cmp     r11, rax
        jbe     3f
        mov     qword ptr [rcx + r10*8], r11    // shift up
        dec     r10
        jmp     2b
3:      mov     qword ptr [rcx + r10*8], rax
        inc     r9
        cmp     r9, r8
        jb      1b
        jmp     .Lsk_ret

.Lsk_radix:
        mov     rsi, rcx                        // src
        mov     rdi, rdx                        // dst
        mov     r12, rcx                        // the original array
        // zero the 8 histograms
        lea     rbx, [rip + sp_hist]
        vxorps  ymm0, ymm0, ymm0
        xor     eax, eax
4:      vmovups ymmword ptr [rbx + rax], ymm0
        add     rax, 32
        cmp     rax, 8*256*4
        jb      4b
        // count all 8 digits in one pass
        xor     r9d, r9d
5:      mov     rax, qword ptr [rsi + r9*8]
        .irp    d, 0, 1, 2, 3, 4, 5, 6, 7
        movzx   r10d, al
        inc     dword ptr [rbx + r10*4 + \d*1024]
        shr     rax, 8
        .endr
        inc     r9
        cmp     r9, r8
        jb      5b
        // one stable scatter pass per digit
        xor     r13d, r13d                      // digit number d
.Lsk_pass:
        mov     r14, r13
        shl     r14, 10
        add     r14, rbx                        // hist of digit d
        // skip a digit every key shares (one bucket holds all n)
        xor     eax, eax
6:      cmp     dword ptr [r14 + rax*4], r8d
        je      .Lsk_next
        inc     eax
        cmp     eax, 256
        jb      6b
        // exclusive prefix sums -> start offsets
        xor     eax, eax
        xor     r10d, r10d
7:      mov     r11d, dword ptr [r14 + rax*4]
        mov     dword ptr [r14 + rax*4], r10d
        add     r10d, r11d
        inc     eax
        cmp     eax, 256
        jb      7b
        // scatter src -> dst
        lea     rcx, [r13*8]                    // shift = 8*d
        xor     r9d, r9d
8:      mov     rax, qword ptr [rsi + r9*8]
        mov     rdx, rax
        shr     rdx, cl
        movzx   edx, dl
        mov     r10d, dword ptr [r14 + rdx*4]
        mov     qword ptr [rdi + r10*8], rax
        inc     r10d
        mov     dword ptr [r14 + rdx*4], r10d
        inc     r9
        cmp     r9, r8
        jb      8b
        xchg    rsi, rdi
.Lsk_next:
        inc     r13
        cmp     r13, 8
        jb      .Lsk_pass
        cmp     rsi, r12                        // result in tmp? copy back
        je      .Lsk_ret
        xor     r9d, r9d
9:      mov     rax, qword ptr [rsi + r9*8]
        mov     qword ptr [r12 + r9*8], rax
        inc     r9
        cmp     r9, r8
        jb      9b
.Lsk_ret:
        pop     r14
        pop     r13
        pop     r12
        pop     rdi
        pop     rsi
        pop     rbx
        vzeroupper
        ret

// ---------------------------------------------------------------------------
// build_keys (internal leaf): keys[n++] = (~bits(p_j) << 32) | j for every j < V
// with exponent field of p_j >= e_low (e_low = 0: all). Returns rax = n.
//   rcx = p, rdx = V, r8 = keys, r9d = e_low.
// ---------------------------------------------------------------------------
        .p2align 4
build_keys:
        xor     eax, eax                        // n
        xor     r10d, r10d                      // j
        shl     r9d, 23                         // bits >= e_low<<23  <=>  field >= e_low
        test    rdx, rdx
        jz      3f
1:      mov     r11d, dword ptr [rcx + r10*4]
        cmp     r11d, r9d
        jb      2f
        not     r11d
        shl     r11, 32
        or      r11, r10
        mov     qword ptr [r8 + rax*8], r11
        inc     rax
2:      inc     r10
        cmp     r10, rdx
        jb      1b
3:      ret

// ---------------------------------------------------------------------------
// keep_count (internal leaf): sampler.c keep_count over sorted keys.
//   rcx = keys, rdx = n, r8 = V, r9 = top_k, xmm3 = top_p
//   returns rax = keep, r10 = decided (0/1).  Clobbers r11, xmm0, xmm1.
// ---------------------------------------------------------------------------
        .p2align 4
keep_count:
        mov     rax, r8                         // keep = V
        test    r9, r9
        jle     1f
        cmp     r9, rax
        jge     1f
        mov     rax, r9                         // keep = top_k
1:      mov     r10d, 1                         // decided
        vxorps  xmm0, xmm0, xmm0
        vucomiss xmm3, xmm0                     // top_p > 0 ?
        jbe     .Lkc_nop
        vucomiss xmm3, dword ptr [rip + S_ONE]  // top_p < 1 ?
        jae     .Lkc_nop
        // cum over the first min(keep, n) sorted entries
        mov     r8, rax                         // lim = min(keep, n)
        cmp     r8, rdx
        jbe     2f
        mov     r8, rdx
2:      xor     r11d, r11d                      // kp
3:      cmp     r11, r8
        jae     4f                              // not broke
        mov     r9, qword ptr [rcx + r11*8]     // key i (top_k no longer needed)
        shr     r9, 32
        not     r9d                             // bits of p_i
        vmovd   xmm1, r9d
        vaddss  xmm0, xmm0, xmm1                // cum += p_i
        inc     r11                             // kp++
        vucomiss xmm0, xmm3                     // cum >= top_p ?
        jae     5f                              // broke
        jmp     3b
4:      cmp     rax, rdx                        // !broke && keep > n -> undecided
        jbe     5f
        xor     r10d, r10d
5:      mov     rax, r11                        // keep = kp
        jmp     .Lkc_min1
.Lkc_nop:
        cmp     rax, rdx                        // no top-p: keep > n -> undecided
        jbe     .Lkc_min1
        xor     r10d, r10d
.Lkc_min1:
        test    rax, rax
        jg      6f
        mov     eax, 1
6:      ret

// ---------------------------------------------------------------------------
// i64 sample_token(const float* l, i64 V, float temperature, i64 top_k,
//                  float top_p, Rng* rng)
//   args: rcx = l, rdx = V, xmm2 = temperature, r9 = top_k,
//         [rsp+40] = top_p (f32), [rsp+48] = rng
//
// Frame: 8 pushes + 104 (8 + 64 + 104 = 176): [rsp+32..] locals,
//   [rsp+48] xmm6 save, [rsp+64] xmm7 save
// Registers: rbx = l, rbp = V, r12 = top_k, r13 = rng, r14 = p, r15 = n,
//            rsi = keys, rdi = keep; xmm6 = top_p, xmm7 = temperature / sum
// ---------------------------------------------------------------------------
#define ST_FRAME 104
#define ST_ARG5  (ST_FRAME + 64 + 8 + 32)       // [rsp + ST_ARG5] = top_p at entry+40
#define ST_ARG6  (ST_FRAME + 64 + 8 + 40)
#define ST_USEK  32
#define ST_USEP  36
#define ST_SUM   40

        .globl  sample_token
        .p2align 4
sample_token:
        // greedy check first: !(temperature > 0)  (C: temperature <= 0.0f)
        vxorps  xmm0, xmm0, xmm0
        vucomiss xmm0, xmm2                     // 0 >= T (ordered)?
        jae     sample_argmax                   // tail call: same rcx, rdx
        push    rbx
        push    rbp
        push    rsi
        push    rdi
        push    r12
        push    r13
        push    r14
        push    r15
        sub     rsp, ST_FRAME
        vmovaps xmmword ptr [rsp + 48], xmm6
        vmovaps xmmword ptr [rsp + 64], xmm7
        mov     rbx, rcx
        mov     rbp, rdx
        mov     r12, r9
        vmovss  xmm6, dword ptr [rsp + ST_ARG5]
        mov     r13, qword ptr [rsp + ST_ARG6]
        vmovaps xmm7, xmm2
        mov     rcx, rbp
        call    sp_reserve
        test    eax, eax
        jnz     .Lst_oom
        mov     r14, qword ptr [rip + sp_p]
        mov     rsi, qword ptr [rip + sp_keys]

        // ---- max_l ----
        mov     rcx, rbx
        mov     rdx, rbp
        call    vec_max
        // ---- p_j = (l_j - max) / T  (vector, exact per element) ----
        vbroadcastss ymm1, xmm0                 // max
        vbroadcastss ymm2, xmm7                 // T
        xor     eax, eax
        mov     r8, rbp
        and     r8, -8
        jz      2f
1:      vmovups ymm0, ymmword ptr [rbx + rax*4]
        vsubps  ymm0, ymm0, ymm1
        vdivps  ymm0, ymm0, ymm2
        vmovups ymmword ptr [r14 + rax*4], ymm0
        add     rax, 8
        cmp     rax, r8
        jb      1b
2:      cmp     rax, rbp
        jae     3f
        vmovss  xmm0, dword ptr [rbx + rax*4]
        vsubss  xmm0, xmm0, xmm1
        vdivss  xmm0, xmm0, xmm2
        vmovss  dword ptr [r14 + rax*4], xmm0
        inc     rax
        jmp     2b
3:      // ---- e_j = expf(...) in place, bit-exact (ops.S) ----
        mov     rcx, r14
        mov     rdx, r14
        mov     r8, rbp
        call    expf_vec
        // ---- sum, strictly sequential in index order (float) ----
        vxorps  xmm0, xmm0, xmm0
        xor     eax, eax
        mov     r8, rbp
        and     r8, -4
        jz      5f
4:      vaddss  xmm0, xmm0, dword ptr [r14 + rax*4]
        vaddss  xmm0, xmm0, dword ptr [r14 + rax*4 + 4]
        vaddss  xmm0, xmm0, dword ptr [r14 + rax*4 + 8]
        vaddss  xmm0, xmm0, dword ptr [r14 + rax*4 + 12]
        add     rax, 4
        cmp     rax, r8
        jb      4b
5:      cmp     rax, rbp
        jae     6f
        vaddss  xmm0, xmm0, dword ptr [r14 + rax*4]
        inc     rax
        jmp     5b
6:      vmovss  dword ptr [rsp + ST_SUM], xmm0
        // ---- p_j = e_j / sum ----
        vbroadcastss ymm1, xmm0
        xor     eax, eax
        mov     r8, rbp
        and     r8, -8
        jz      8f
7:      vmovups ymm0, ymmword ptr [r14 + rax*4]
        vdivps  ymm0, ymm0, ymm1
        vmovups ymmword ptr [r14 + rax*4], ymm0
        add     rax, 8
        cmp     rax, r8
        jb      7b
8:      cmp     rax, rbp
        jae     9f
        vmovss  xmm0, dword ptr [r14 + rax*4]
        vdivss  xmm0, xmm0, xmm1
        vmovss  dword ptr [r14 + rax*4], xmm0
        inc     rax
        jmp     8b
9:
        // ---- use_k / use_p ----
        xor     eax, eax
        test    r12, r12
        jle     10f
        cmp     r12, rbp
        jge     10f
        mov     eax, 1
10:     mov     dword ptr [rsp + ST_USEK], eax
        xor     eax, eax
        vxorps  xmm0, xmm0, xmm0
        vucomiss xmm6, xmm0
        jbe     11f
        vucomiss xmm6, dword ptr [rip + S_ONE]
        jae     11f
        mov     eax, 1
11:     mov     dword ptr [rsp + ST_USEP], eax

        // ---- candidate threshold (exponent histogram) ----
        xor     r9d, r9d                        // e_low = 0 (all)
        mov     eax, dword ptr [rsp + ST_USEK]
        or      eax, dword ptr [rsp + ST_USEP]
        jz      .Lst_build
        mov     eax, dword ptr [rsp + ST_SUM]   // isfinite(sum) && sum > 0
        test    eax, eax
        jle     .Lst_build                      // sign set or zero
        cmp     eax, 0x7f800000
        jae     .Lst_build                      // inf / NaN
        call    pick_threshold                  // -> r9d = e_low (0 = none)
.Lst_build:
        mov     rcx, r14
        mov     rdx, rbp
        mov     r8, rsi
        call    build_keys
        mov     r15, rax                        // n candidates
        mov     rcx, rsi
        mov     rdx, qword ptr [rip + sp_tmp]
        mov     r8, r15
        call    sort_keys
        mov     rcx, rsi
        mov     rdx, r15
        mov     r8, rbp
        mov     r9, r12
        vmovaps xmm3, xmm6
        call    keep_count
        test    r10, r10
        jnz     .Lst_draw
        // the candidates did not cover the cut: sort everything
        mov     rcx, r14
        mov     rdx, rbp
        mov     r8, rsi
        xor     r9d, r9d
        call    build_keys
        mov     r15, rax
        mov     rcx, rsi
        mov     rdx, qword ptr [rip + sp_tmp]
        mov     r8, r15
        call    sort_keys
        mov     rcx, rsi
        mov     rdx, r15
        mov     r8, rbp
        mov     r9, r12
        vmovaps xmm3, xmm6
        call    keep_count
.Lst_draw:
        mov     rdi, rax                        // keep
        // z = sum of the kept p, in sorted order
        vxorps  xmm7, xmm7, xmm7
        xor     ecx, ecx
12:     mov     rax, qword ptr [rsi + rcx*8]
        shr     rax, 32
        not     eax
        vmovd   xmm0, eax
        vaddss  xmm7, xmm7, xmm0
        inc     rcx
        cmp     rcx, rdi
        jb      12b
        mov     rcx, r13
        call    rng_uniform
        vmulss  xmm1, xmm0, xmm7                // u = rng_uniform * z
        mov     rax, qword ptr [rsi + rdi*8 - 8]
        mov     eax, eax                        // chosen = id of the last kept
        vxorps  xmm2, xmm2, xmm2                // acc
        xor     ecx, ecx
13:     mov     rdx, qword ptr [rsi + rcx*8]
        mov     r8, rdx
        shr     r8, 32
        not     r8d
        vmovd   xmm0, r8d
        vaddss  xmm2, xmm2, xmm0                // acc += p_i
        vucomiss xmm2, xmm1                     // u <= acc ?
        jae     14f
        inc     rcx
        cmp     rcx, rdi
        jb      13b
        jmp     .Lst_ret
14:     mov     eax, edx                        // chosen = id_i
.Lst_ret:
        vmovaps xmm6, xmmword ptr [rsp + 48]
        vmovaps xmm7, xmmword ptr [rsp + 64]
        add     rsp, ST_FRAME
        pop     r15
        pop     r14
        pop     r13
        pop     r12
        pop     rdi
        pop     rsi
        pop     rbp
        pop     rbx
        vzeroupper
        ret
.Lst_oom:
        mov     rcx, rbx                        // no memory: greedy
        mov     rdx, rbp
        call    sample_argmax
        jmp     .Lst_ret

// ---------------------------------------------------------------------------
// pick_threshold (internal, called from sample_token with its registers):
//   in: r14 = p, rbp = V, r12 = top_k, xmm6 = top_p, [rsp+8+ST_USEK/USEP] flags
//   out: r9d = e_low (0 = no usable threshold).
// Histogram of exponent fields: cnt[e], mass[e] (double). Four interleaved copies
// (element j goes to copy j&3) break the store-to-load dependency through one
// bucket; the copies are combined when scanning. The mass sums differ from the C
// code's single sum only in rounding order -- that can only move the threshold,
// never the result (keep_count re-checks the cut and falls back to a full sort).
// Then, from the top exponent down: c += cnt[e], ms += mass[e]; stop at the first
// e with c >= need (top_k, or V) or (use_p and ms >= top_p + 1e-3).
// ---------------------------------------------------------------------------
        .p2align 4
pick_threshold:
        lea     r10, [rip + sp_cnt]
        lea     r11, [rip + sp_mass]
        vxorps  ymm0, ymm0, ymm0
        xor     eax, eax
1:      vmovups ymmword ptr [r10 + rax], ymm0
        add     rax, 32
        cmp     rax, 256*4*4
        jb      1b
        xor     eax, eax
2:      vmovups ymmword ptr [r11 + rax], ymm0
        add     rax, 32
        cmp     rax, 256*8*4
        jb      2b
        // histogram, 4 elements per iteration into the 4 copies
        xor     eax, eax
        mov     r8, rbp
        and     r8, -4
        jz      4f
3:      .irp    c, 0, 1, 2, 3
        mov     edx, dword ptr [r14 + rax*4 + \c*4]
        shr     edx, 23
        movzx   edx, dl                         // exponent field (p >= 0: no sign)
        inc     dword ptr [r10 + rdx*4 + \c*1024]
        vcvtss2sd xmm1, xmm1, dword ptr [r14 + rax*4 + \c*4]
        vaddsd  xmm1, xmm1, qword ptr [r11 + rdx*8 + \c*2048]
        vmovsd  qword ptr [r11 + rdx*8 + \c*2048], xmm1
        .endr
        add     rax, 4
        cmp     rax, r8
        jb      3b
4:      cmp     rax, rbp
        jae     5f
        mov     edx, dword ptr [r14 + rax*4]
        shr     edx, 23
        movzx   edx, dl
        inc     dword ptr [r10 + rdx*4]
        vcvtss2sd xmm1, xmm1, dword ptr [r14 + rax*4]
        vaddsd  xmm1, xmm1, qword ptr [r11 + rdx*8]
        vmovsd  qword ptr [r11 + rdx*8], xmm1
        inc     rax
        jmp     4b
5:      // need = use_k ? top_k : V
        mov     rcx, rbp
        cmp     dword ptr [rsp + 8 + ST_USEK], 0
        je      6f
        mov     rcx, r12
6:      vcvtss2sd xmm2, xmm2, xmm6              // (double)top_p + 1e-3
        vaddsd  xmm2, xmm2, qword ptr [rip + S_MARGIN]
        vxorpd  xmm3, xmm3, xmm3                // ms
        xor     r8d, r8d                        // c
        mov     edx, 255                        // e
7:      mov     eax, dword ptr [r10 + rdx*4]
        add     eax, dword ptr [r10 + rdx*4 + 1024]
        add     eax, dword ptr [r10 + rdx*4 + 2048]
        add     eax, dword ptr [r10 + rdx*4 + 3072]
        add     r8, rax
        vaddsd  xmm3, xmm3, qword ptr [r11 + rdx*8]
        vaddsd  xmm3, xmm3, qword ptr [r11 + rdx*8 + 2048]
        vaddsd  xmm3, xmm3, qword ptr [r11 + rdx*8 + 4096]
        vaddsd  xmm3, xmm3, qword ptr [r11 + rdx*8 + 6144]
        cmp     r8, rcx
        jge     8f
        cmp     dword ptr [rsp + 8 + ST_USEP], 0
        je      9f
        vucomisd xmm3, xmm2
        jae     8f
9:      dec     edx
        jnz     7b
        xor     r9d, r9d                        // never covered: no threshold
        ret
8:      mov     r9d, edx
        ret

// ---------------------------------------------------------------------------
// vec_max (internal leaf): xmm0 = max of l[0..n)  (n >= 1).  rcx = l, rdx = n.
// ---------------------------------------------------------------------------
        .p2align 4
vec_max:
        vbroadcastss ymm0, dword ptr [rcx]
        mov     r8, rdx
        and     r8, -8
        xor     eax, eax
        test    r8, r8
        jz      2f
1:      vmaxps  ymm0, ymm0, ymmword ptr [rcx + rax*4]
        add     rax, 8
        cmp     rax, r8
        jb      1b
2:      vextractf128 xmm1, ymm0, 1
        vmaxps  xmm0, xmm0, xmm1
        vpermilps xmm1, xmm0, 0x4E
        vmaxps  xmm0, xmm0, xmm1
        vpermilps xmm1, xmm0, 0xB1
        vmaxps  xmm0, xmm0, xmm1
3:      cmp     rax, rdx
        jae     4f
        vmaxss  xmm0, xmm0, dword ptr [rcx + rax*4]
        inc     rax
        jmp     3b
4:      ret
