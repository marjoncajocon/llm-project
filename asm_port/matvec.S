// matvec.S - weight-matrix kernels: y = x @ W (owner: kernels).
//
// Exports (Win64 ABI; see DESIGN.md, Mat layout in model.inc)
//   void* mat_q8_pack(Mat* m)
//         Build the VNNI-friendly packed copy of a Q8 matrix (m->data int8 (K,N)
//         row-major). Sets m->packed, m->colsum, m->aux (= Kpad). Returns packed
//         (0 if the allocation failed).
//   void* mat_q8_from_f32(Mat* out, const float* w, i64 K, i64 N)
//         Per-column int8 quantization, bit-identical to quantize_int8_percol()
//         in ../quant.c; fills out (kind Q8, data, scale) then packs it.
//   void  matvec(const Mat* m, const float* x, float* y)
//         y[0..N) = x[0..K) @ W. Q8 is bit-exact with qmatmul() in ../quant.c.
//   void  matmul_rows(const Mat* m, const float* X, i64 M, float* Y)
//         M rows at once (prefill): Y[i,:] = X[i,:] @ W, row i bit-identical
//         to matvec(m, X[i,:]) (matvec IS matmul_rows with M = 1).
//   tuning knobs (i64 data, read on every call; the test may change them):
//         mv_small_bytes        weight bytes below which a call runs single-thread
//         mv_chunks_per_thread  target number of work chunks per pool thread
//         mv_min_chunk_bytes    lower bound for the bytes of one chunk
//
// ---------------------------------------------------------------------------
// Q8 packed layout (built by mat_q8_pack)
//   Kpad = roundup(K,16), Npad = roundup(N,16). The matrix is cut into PANELS
//   of 16 output columns; panel p is one contiguous block of Kpad*16 bytes, and
//   panels follow each other (so a run of panels = one sequential stream).
//   Inside a panel, k is walked in groups of 4 ("kgroups"); one kgroup is one
//   64-byte cache line:
//        bytes  0..31 : columns 0..7  of the panel, 4 bytes each = w[4g+0..3][col]
//        bytes 32..63 : columns 8..15 of the panel
//   so  packed[p*Kpad*16 + g*64 + h*32 + c*4 + t] = W[4g+t][16p + 8h + c]
//   (zero for k >= K or column >= N). This is exactly the operand shape of
//   vpdpbusd ymm: each dword lane j accumulates sum_t u8(x[4g+t]) * s8(w[..][j]),
//   so one instruction does 8 columns x 4 k (32 MACs); x[4g..4g+3] is broadcast
//   with vpbroadcastd.
//   colsum (int32[Npad]) holds 128 * sum_k W[k][j]: the VNNI kernel feeds
//   activations as u8 = q + 128 (q in [-127,127] -> [1,255]) and subtracts
//   128*colsum afterwards, which leaves the exact int32 dot product.
//   The AVX2-only kernel (no VNNI) uses the same layout with the sign trick:
//   vpmaddubsw(|q|, sign(q)*w) - each i16 pair sum is <= 2*127*127 = 32258, so
//   it never saturates (feeding q+128 instead could reach 2*255*127 = 64770).
//
// Q8 arithmetic (identical to qmatmul): as = max|x|/127 (1 if 0), inv = 1/as,
//   q = cvtps2dq(x*inv) (round-half-even like lrintf) clamped to +-127;
//   acc = exact int32 dot; y[j] = (as*scale[j]) * (float)acc[j].
//
// F32 kinds: every output element is a single FMA chain in increasing k
//   (acc = 0; acc = fma(x[k], w[k][j], acc)) for F32, and a fixed 16-lane
//   split + fixed horizontal reduction for F32T - the same instruction sequence
//   per element whatever the register blocking, so matmul_rows == matvec bitwise.
//
// Threading: work is split over output columns (panels / 16-column units /
// rows of W^T) with pool_run; each chunk is a contiguous run of memory.
// Not reentrant: matvec/matmul_rows use static scratch (call from one thread).

#include "model.inc"
.intel_syntax noprefix

#define XQ_BYTES  (4*1024*1024)         // quantized-activation scratch
#define MAXROWS   4096                  // max rows per Q8 block in matmul_rows

// ctx: the job description shared with the chunk functions (static, 1 job at a time)
#define C_PACKED  0
#define C_KPAD    8
#define C_N       16
#define C_SCALE   24
#define C_COLSUM  32
#define C_XQ      40
#define C_AS      48
#define C_Y       56
#define C_M       64
#define C_X       72
#define C_K       80
#define C_W       88
#define C_XORB    96
#define C_VNNI    104

// ---------------------------------------------------------------------------
        .section .rdata,"dr"
        .balign 64
masktab:        .long -1,-1,-1,-1,-1,-1,-1,-1, 0,0,0,0,0,0,0,0   // [masktab+32-4c] = first c lanes
v_127i:         .long 127,127,127,127,127,127,127,127
v_m127i:        .long -127,-127,-127,-127,-127,-127,-127,-127
v_127f:         .float 127.0,127.0,127.0,127.0,127.0,127.0,127.0,127.0
v_1f:           .float 1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0
v_ones16:       .short 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
permd_idx:      .long 0,4,1,5,2,6,3,7
c_absmask:      .long 0x7fffffff
c_one:          .float 1.0
c_127f:         .float 127.0
        .balign 8
q8r_vnni_tab:   .quad 0, q8r_vnni_dec, q8r_vnni_2, q8r_vnni_3, q8r_vnni_4, q8r_vnni_5, q8r_vnni_6
f32r_tab:       .quad 0, f32r_1, f32r_2, f32r_3, f32r_4, f32r_5, f32r_6

        .data
        .balign 8
        .globl  mv_small_bytes, mv_chunks_per_thread, mv_min_chunk_bytes
mv_small_bytes:       .quad 65536
mv_chunks_per_thread: .quad 4
mv_min_chunk_bytes:   .quad 16384
        .globl  mv_mem_threads, mv_mem_rows
mv_mem_threads:       .quad 0           // threads for memory-bound calls (0 = pool_mem_threads)
mv_mem_rows:          .quad 16          // calls with <= this many rows count as memory-bound

        .bss
        .balign 64
xq_buf: .space  XQ_BYTES                // quantized activation rows (stride Kpad)
as_buf: .space  MAXROWS*4               // per-row activation scales
        .balign 64
ctx:    .space  128

        .text

// ===========================================================================
// Helpers: save/restore the callee-saved xmm6..xmm15 (low 128 bits are
// non-volatile in Win64) at [rsp+off] (16-byte aligned).
// ===========================================================================
.macro SAVE_XMM off
        vmovaps [rsp + \off +   0], xmm6
        vmovaps [rsp + \off +  16], xmm7
        vmovaps [rsp + \off +  32], xmm8
        vmovaps [rsp + \off +  48], xmm9
        vmovaps [rsp + \off +  64], xmm10
        vmovaps [rsp + \off +  80], xmm11
        vmovaps [rsp + \off +  96], xmm12
        vmovaps [rsp + \off + 112], xmm13
        vmovaps [rsp + \off + 128], xmm14
        vmovaps [rsp + \off + 144], xmm15
.endm
.macro RESTORE_XMM off
        vmovaps xmm6,  [rsp + \off +   0]
        vmovaps xmm7,  [rsp + \off +  16]
        vmovaps xmm8,  [rsp + \off +  32]
        vmovaps xmm9,  [rsp + \off +  48]
        vmovaps xmm10, [rsp + \off +  64]
        vmovaps xmm11, [rsp + \off +  80]
        vmovaps xmm12, [rsp + \off +  96]
        vmovaps xmm13, [rsp + \off + 112]
        vmovaps xmm14, [rsp + \off + 128]
        vmovaps xmm15, [rsp + \off + 144]
.endm
// prologue/epilogue for the chunk functions called by pool_run:
// 7 GPR pushes + 192 bytes (locals [rsp..rsp+32), xmm save [rsp+32..rsp+192))
.macro CHUNK_ENTER
        push    rbx
        push    rsi
        push    rdi
        push    r12
        push    r13
        push    r14
        push    r15
        sub     rsp, 192
        SAVE_XMM 32
.endm
.macro CHUNK_LEAVE
        RESTORE_XMM 32
        vzeroupper
        add     rsp, 192
        pop     r15
        pop     r14
        pop     r13
        pop     r12
        pop     rdi
        pop     rsi
        pop     rbx
        ret
.endm

// ===========================================================================
// quant_row (internal): quantize one activation row exactly like qmatmul.
//   in : rcx = x (float*), rdx = K, r8 = out (u8/s8 bytes), r9d = xor byte
//        (0x80 -> u8 = q+128 for VNNI, 0 -> plain s8 for the AVX2 kernel)
//   out: xmm0 = as (activation scale), out[0..K) written
//   clobbers: rax, r10, r11, ymm0-ymm5 (volatile only - no xmm save needed)
// ===========================================================================
quant_row:
        vbroadcastss ymm5, dword ptr [rip + c_absmask]
        vxorps  ymm0, ymm0, ymm0
        vxorps  ymm1, ymm1, ymm1
        xor     eax, eax
        mov     r10, rdx
        and     r10, -16
        jmp     2f
1:      vandps  ymm2, ymm5, [rcx + rax*4]           // |x|, 2 independent max chains
        vmaxps  ymm0, ymm0, ymm2
        vandps  ymm3, ymm5, [rcx + rax*4 + 32]
        vmaxps  ymm1, ymm1, ymm3
        add     rax, 16
2:      cmp     rax, r10
        jb      1b
        vmaxps  ymm0, ymm0, ymm1                    // horizontal max (exact in any order)
        vextractf128 xmm1, ymm0, 1
        vmaxps  xmm0, xmm0, xmm1
        vmovhlps xmm1, xmm0, xmm0
        vmaxps  xmm0, xmm0, xmm1
        vmovshdup xmm1, xmm0
        vmaxss  xmm0, xmm0, xmm1
        jmp     4f
3:      vmovss  xmm2, dword ptr [rcx + rax*4]
        vandps  xmm2, xmm2, xmm5
        vmaxss  xmm0, xmm0, xmm2
        inc     rax
4:      cmp     rax, rdx
        jb      3b
        // as = mx > 0 ? mx/127 : 1 ; inv = 1/as  (true divisions, as in C)
        vmovss  xmm3, dword ptr [rip + c_one]
        vxorps  xmm1, xmm1, xmm1
        vucomiss xmm0, xmm1
        jbe     5f
        vdivss  xmm3, xmm0, dword ptr [rip + c_127f]
5:      vmovd   r11d, xmm3                          // keep as in r11d
        vmovss  xmm4, dword ptr [rip + c_one]
        vdivss  xmm4, xmm4, xmm3
        vbroadcastss ymm4, xmm4                     // ymm4 = inv
        vmovd   xmm5, r9d
        vpbroadcastb ymm5, xmm5                     // ymm5 = xor byte
        xor     eax, eax
        mov     r10, rdx
        and     r10, -32
        jmp     7f
6:      vmulps  ymm0, ymm4, [rcx + rax*4]           // 32 floats -> 32 bytes
        vmulps  ymm1, ymm4, [rcx + rax*4 + 32]
        vmulps  ymm2, ymm4, [rcx + rax*4 + 64]
        vmulps  ymm3, ymm4, [rcx + rax*4 + 96]
        vcvtps2dq ymm0, ymm0                        // round half to even (MXCSR default)
        vcvtps2dq ymm1, ymm1
        vcvtps2dq ymm2, ymm2
        vcvtps2dq ymm3, ymm3
        vpminsd ymm0, ymm0, [rip + v_127i]
        vpminsd ymm1, ymm1, [rip + v_127i]
        vpminsd ymm2, ymm2, [rip + v_127i]
        vpminsd ymm3, ymm3, [rip + v_127i]
        vpmaxsd ymm0, ymm0, [rip + v_m127i]
        vpmaxsd ymm1, ymm1, [rip + v_m127i]
        vpmaxsd ymm2, ymm2, [rip + v_m127i]
        vpmaxsd ymm3, ymm3, [rip + v_m127i]
        vpackssdw ymm0, ymm0, ymm1                  // per lane: a0-3 b0-3 | a4-7 b4-7
        vpackssdw ymm2, ymm2, ymm3
        vpacksswb ymm0, ymm0, ymm2                  // dwords: a0 b0 c0 d0 a1 b1 c1 d1
        vmovdqu ymm1, [rip + permd_idx]
        vpermd  ymm0, ymm1, ymm0                    // -> a b c d in order
        vpxor   ymm0, ymm0, ymm5
        vmovdqu [r8 + rax], ymm0
        add     rax, 32
7:      cmp     rax, r10
        jb      6b
        jmp     11f
8:      vmulss  xmm0, xmm4, dword ptr [rcx + rax*4] // scalar tail
        vcvtss2si r10d, xmm0
        cmp     r10d, 127
        jle     9f
        mov     r10d, 127
9:      cmp     r10d, -127
        jge     10f
        mov     r10d, -127
10:     xor     r10d, r9d
        mov     [r8 + rax], r10b
        inc     rax
11:     cmp     rax, rdx
        jb      8b
        vmovd   xmm0, r11d
        ret

// ===========================================================================
// epi16 / epi16_nocs (internal): finish one row x one 16-column panel.
//   in : ymm12/ymm13 = int32 dots for columns j0..j0+7 / j0+8..j0+15
//        (epi16 first subtracts 128*colsum - VNNI offset; _nocs does not),
//        ymm14 = as broadcast, rax = &Y_row[0], r13 = j0, rbx = ctx
//   out: Y_row[j0 .. min(j0+16,N)) = (as*scale[j]) * (float)dot[j]
//   clobbers: rax, rcx, rdx, ymm12, ymm13, ymm15
// ===========================================================================
epi16:
        mov     rdx, [rbx + C_COLSUM]
        vpsubd  ymm12, ymm12, [rdx + r13*4]
        vpsubd  ymm13, ymm13, [rdx + r13*4 + 32]
epi16_nocs:
        vcvtdq2ps ymm12, ymm12
        vcvtdq2ps ymm13, ymm13
        mov     rdx, [rbx + C_SCALE]
        mov     rcx, [rbx + C_N]
        sub     rcx, r13
        cmp     rcx, 16
        jb      1f
        vmulps  ymm15, ymm14, [rdx + r13*4]         // as*scale[j]  (same order as C)
        vmulps  ymm12, ymm15, ymm12                 // * (float)acc
        vmovups [rax + r13*4], ymm12
        vmulps  ymm15, ymm14, [rdx + r13*4 + 32]
        vmulps  ymm13, ymm15, ymm13
        vmovups [rax + r13*4 + 32], ymm13
        ret
1:      sub     rsp, 72                             // partial last panel: scalar
        vmovups [rsp], ymm12
        vmovups [rsp + 32], ymm13
        lea     rax, [rax + r13*4]
        lea     rdx, [rdx + r13*4]
2:      dec     rcx
        vmovss  xmm15, dword ptr [rdx + rcx*4]
        vmulss  xmm15, xmm14, xmm15
        vmulss  xmm15, xmm15, dword ptr [rsp + rcx*4]
        vmovss  dword ptr [rax + rcx*4], xmm15
        test    rcx, rcx
        jnz     2b
        add     rsp, 72
        ret

// ROWPTR off: ymm14 = as[r14+off]; rax = &Y[(r14+off)*N]
.macro ROWPTR off
        mov     rax, [rbx + C_AS]
        vbroadcastss ymm14, dword ptr [rax + r14*4 + 4*\off]
        lea     rax, [r14 + \off]
        imul    rax, [rbx + C_N]
        shl     rax, 2
        add     rax, [rbx + C_Y]
.endm

// ===========================================================================
// Q8 panel kernels (internal). Common convention:
//   in : r10 = panel base (packed + p*Kpad*16), r11 = xq row r14, r9 = Kpad,
//        r13 = j0 = 16p, r14 = first row index m, rbx = ctx
//   clobbers: rax, rcx, rdx, rsi, rdi, r8, r12, r15, ymm0-ymm15
//   preserves: rbx, r9, r10, r11, r13, r14
// ===========================================================================

// q8r_vnni_dec: ONE row, the decode kernel. 4 kgroups (16 k, 256 bytes) per
// iteration into 8 independent accumulators (hides vpdpbusd's 5-cycle latency).
q8r_vnni_dec:
        mov     rdx, r10
        xor     eax, eax
        vpxor   ymm0, ymm0, ymm0
        vpxor   ymm1, ymm1, ymm1
        vpxor   ymm2, ymm2, ymm2
        vpxor   ymm3, ymm3, ymm3
        vpxor   ymm4, ymm4, ymm4
        vpxor   ymm5, ymm5, ymm5
        vpxor   ymm6, ymm6, ymm6
        vpxor   ymm7, ymm7, ymm7
        .p2align 5
1:      vpbroadcastd ymm8,  dword ptr [r11 + rax]
        vpbroadcastd ymm9,  dword ptr [r11 + rax + 4]
        vpbroadcastd ymm10, dword ptr [r11 + rax + 8]
        vpbroadcastd ymm11, dword ptr [r11 + rax + 12]
        {vex} vpdpbusd ymm0, ymm8,  ymmword ptr [rdx]
        {vex} vpdpbusd ymm1, ymm8,  ymmword ptr [rdx + 32]
        {vex} vpdpbusd ymm2, ymm9,  ymmword ptr [rdx + 64]
        {vex} vpdpbusd ymm3, ymm9,  ymmword ptr [rdx + 96]
        {vex} vpdpbusd ymm4, ymm10, ymmword ptr [rdx + 128]
        {vex} vpdpbusd ymm5, ymm10, ymmword ptr [rdx + 160]
        {vex} vpdpbusd ymm6, ymm11, ymmword ptr [rdx + 192]
        {vex} vpdpbusd ymm7, ymm11, ymmword ptr [rdx + 224]
        add     rdx, 256
        add     rax, 16
        cmp     rax, r9
        jb      1b
        vpaddd  ymm0, ymm0, ymm2
        vpaddd  ymm4, ymm4, ymm6
        vpaddd  ymm12, ymm0, ymm4
        vpaddd  ymm1, ymm1, ymm3
        vpaddd  ymm5, ymm5, ymm7
        vpaddd  ymm13, ymm1, ymm5
        ROWPTR  0
        jmp     epi16

// Q8R_VNNI R: R rows (2..6) share every weight load: per kgroup, load the
// 64-byte line once (ymm14/15) and do 2R vpdpbusd. Accumulators: row r in
// ymm(2r), ymm(2r+1). Row pointers: r11, r12, r15, rdi, rsi, rcx (+rax = k).
.macro Q8R_VNNI R
q8r_vnni_\R:
        lea     r12, [r11 + r9]
        lea     r15, [r11 + r9*2]
        lea     rdi, [r12 + r9*2]
        lea     rsi, [r15 + r9*2]
        lea     rcx, [rdi + r9*2]
        vpxor   ymm0, ymm0, ymm0
        vpxor   ymm1, ymm1, ymm1
        vpxor   ymm2, ymm2, ymm2
        vpxor   ymm3, ymm3, ymm3
        vpxor   ymm4, ymm4, ymm4
        vpxor   ymm5, ymm5, ymm5
        vpxor   ymm6, ymm6, ymm6
        vpxor   ymm7, ymm7, ymm7
        vpxor   ymm8, ymm8, ymm8
        vpxor   ymm9, ymm9, ymm9
        vpxor   ymm10, ymm10, ymm10
        vpxor   ymm11, ymm11, ymm11
        mov     rdx, r10
        xor     eax, eax
        .p2align 5
1:      vmovdqu ymm14, [rdx]
        vmovdqu ymm15, [rdx + 32]
        vpbroadcastd ymm12, dword ptr [r11 + rax]
        {vex} vpdpbusd ymm0, ymm12, ymm14
        {vex} vpdpbusd ymm1, ymm12, ymm15
        vpbroadcastd ymm13, dword ptr [r12 + rax]
        {vex} vpdpbusd ymm2, ymm13, ymm14
        {vex} vpdpbusd ymm3, ymm13, ymm15
.if \R >= 3
        vpbroadcastd ymm12, dword ptr [r15 + rax]
        {vex} vpdpbusd ymm4, ymm12, ymm14
        {vex} vpdpbusd ymm5, ymm12, ymm15
.endif
.if \R >= 4
        vpbroadcastd ymm13, dword ptr [rdi + rax]
        {vex} vpdpbusd ymm6, ymm13, ymm14
        {vex} vpdpbusd ymm7, ymm13, ymm15
.endif
.if \R >= 5
        vpbroadcastd ymm12, dword ptr [rsi + rax]
        {vex} vpdpbusd ymm8, ymm12, ymm14
        {vex} vpdpbusd ymm9, ymm12, ymm15
.endif
.if \R >= 6
        vpbroadcastd ymm13, dword ptr [rcx + rax]
        {vex} vpdpbusd ymm10, ymm13, ymm14
        {vex} vpdpbusd ymm11, ymm13, ymm15
.endif
        add     rdx, 64
        add     rax, 4
        cmp     rax, r9
        jb      1b
        vmovdqa ymm12, ymm0
        vmovdqa ymm13, ymm1
        ROWPTR  0
        call    epi16
        vmovdqa ymm12, ymm2
        vmovdqa ymm13, ymm3
        ROWPTR  1
        call    epi16
.if \R >= 3
        vmovdqa ymm12, ymm4
        vmovdqa ymm13, ymm5
        ROWPTR  2
        call    epi16
.endif
.if \R >= 4
        vmovdqa ymm12, ymm6
        vmovdqa ymm13, ymm7
        ROWPTR  3
        call    epi16
.endif
.if \R >= 5
        vmovdqa ymm12, ymm8
        vmovdqa ymm13, ymm9
        ROWPTR  4
        call    epi16
.endif
.if \R >= 6
        vmovdqa ymm12, ymm10
        vmovdqa ymm13, ymm11
        ROWPTR  5
        call    epi16
.endif
        ret
.endm
        Q8R_VNNI 2
        Q8R_VNNI 3
        Q8R_VNNI 4
        Q8R_VNNI 5
        Q8R_VNNI 6

// q8r_avx2_1: ONE row without VNNI (s8 activations). Per kgroup:
//   b = bcast(q[4g..4g+3]); |b|; w' = sign(b)*w (vpsignb); vpmaddubsw(|b|, w')
//   -> i16 pairs (<= 32258, no saturation); vpmaddwd with 1s -> i32; add.
.macro KG_AVX2 g, accA, accB
        vpbroadcastd ymm8, dword ptr [r11 + rax + 4*\g]
        vpabsb  ymm9, ymm8
        vmovdqu ymm10, [rdx + 64*\g]
        vmovdqu ymm11, [rdx + 64*\g + 32]
        vpsignb ymm10, ymm10, ymm8
        vpsignb ymm11, ymm11, ymm8
        vpmaddubsw ymm10, ymm9, ymm10
        vpmaddubsw ymm11, ymm9, ymm11
        vpmaddwd ymm10, ymm10, ymm15
        vpmaddwd ymm11, ymm11, ymm15
        vpaddd  \accA, \accA, ymm10
        vpaddd  \accB, \accB, ymm11
.endm
q8r_avx2_1:
        vmovdqu ymm15, [rip + v_ones16]
        mov     rdx, r10
        xor     eax, eax
        vpxor   ymm0, ymm0, ymm0
        vpxor   ymm1, ymm1, ymm1
        vpxor   ymm2, ymm2, ymm2
        vpxor   ymm3, ymm3, ymm3
        vpxor   ymm4, ymm4, ymm4
        vpxor   ymm5, ymm5, ymm5
        vpxor   ymm6, ymm6, ymm6
        vpxor   ymm7, ymm7, ymm7
        .p2align 5
1:      KG_AVX2 0, ymm0, ymm1
        KG_AVX2 1, ymm2, ymm3
        KG_AVX2 2, ymm4, ymm5
        KG_AVX2 3, ymm6, ymm7
        add     rdx, 256
        add     rax, 16
        cmp     rax, r9
        jb      1b
        vpaddd  ymm0, ymm0, ymm2
        vpaddd  ymm4, ymm4, ymm6
        vpaddd  ymm12, ymm0, ymm4
        vpaddd  ymm1, ymm1, ymm3
        vpaddd  ymm5, ymm5, ymm7
        vpaddd  ymm13, ymm1, ymm5
        ROWPTR  0
        jmp     epi16_nocs

// ===========================================================================
// q8_chunk(ctx, p0, p1): pool_run worker for Q8 - panels [p0,p1), all M rows.
//   Loop order: panel outer, rows inner (the panel, Kpad*16 bytes, stays in
//   L1/L2 while 6-row blocks of activations stream past it).
//   locals: [rsp] = p, [rsp+8] = p1, [rsp+16] = R of the current call
// ===========================================================================
q8_chunk:
        CHUNK_ENTER
        mov     rbx, rcx
        mov     [rsp], rdx
        mov     [rsp + 8], r8
        mov     r9, [rbx + C_KPAD]
.Lq8c_p:
        mov     rax, [rsp]
        cmp     rax, [rsp + 8]
        jae     .Lq8c_done
        mov     r13, rax
        shl     r13, 4                      // j0 = 16p
        mov     r10, r9
        imul    r10, r13
        add     r10, [rbx + C_PACKED]       // panel base
        xor     r14d, r14d                  // m = 0
.Lq8c_m:
        mov     rax, [rbx + C_M]
        sub     rax, r14                    // rows left
        jle     .Lq8c_next
        mov     r11, r14
        imul    r11, r9
        add     r11, [rbx + C_XQ]           // xq row m
        cmp     qword ptr [rbx + C_VNNI], 0
        je      .Lq8c_avx2
        cmp     rax, 6
        jbe     1f
        mov     eax, 6
1:      mov     [rsp + 16], rax
        lea     rcx, [rip + q8r_vnni_tab]
        call    [rcx + rax*8]
        add     r14, [rsp + 16]
        jmp     .Lq8c_m
.Lq8c_avx2:
        call    q8r_avx2_1
        inc     r14
        jmp     .Lq8c_m
.Lq8c_next:
        inc     qword ptr [rsp]
        jmp     .Lq8c_p
.Lq8c_done:
        CHUNK_LEAVE

// ===========================================================================
// qrows_chunk(ctx, r0, r1): quantize activation rows [r0,r1) of the current
// block (X = ctx.X, stride K floats) into xq (stride Kpad) and as[].
// ===========================================================================
qrows_chunk:
        push    rbx
        push    rsi
        push    rdi
        push    r12
        sub     rsp, 40
        mov     rbx, rcx
        mov     rsi, rdx
        mov     rdi, r8
1:      cmp     rsi, rdi
        jae     2f
        mov     rcx, rsi
        imul    rcx, [rbx + C_K]
        shl     rcx, 2
        add     rcx, [rbx + C_X]
        mov     rdx, [rbx + C_K]
        mov     r8, rsi
        imul    r8, [rbx + C_KPAD]
        add     r8, [rbx + C_XQ]
        mov     r9, [rbx + C_XORB]
        call    quant_row
        mov     rax, [rbx + C_AS]
        vmovss  dword ptr [rax + rsi*4], xmm0
        inc     rsi
        jmp     1b
2:      vzeroupper
        add     rsp, 40
        pop     r12
        pop     rdi
        pop     rsi
        pop     rbx
        ret

// ===========================================================================
// F32 (K,N) kernels (internal). Convention:
//   in : r10 = &W[0][j], r11 = &X[m][0], r9 = K*4, r8 = N*4, r13 = j, r14 = m,
//        rbx = ctx
//   clobbers: rax, rcx, rdx, rsi, rdi, r12, r15, ymm0-ymm15
// Every output element: acc = 0; for k = 0..K-1: acc = fma(x[k], w[k][j], acc).
// ===========================================================================

// f32_decw: one row, a strip of rcx units (2..16) of 16 columns (<= 256
// columns = 1 KB per W row, so DRAM sees long contiguous bursts). The strip's
// accumulators live in a 32-byte aligned stack buffer (L1); W rows are
// consumed 4 at a time: acc = fma(x[k],w[k],acc) ... fma(x[k+3],w[k+3],acc),
// i.e. exactly the sequential per-element FMA chain of the F32R kernels.
//   extra in: rcx = units. rbx is used as the strip offset (saved/restored),
//   rbp as the frame pointer (saved/restored).
f32_decw:
        push    rbp
        push    rbx
        mov     rbp, rsp
        sub     rsp, 1024
        and     rsp, -32
        shl     rcx, 6                      // strip bytes (units*16*4)
        mov     r15, rcx
        vxorps  ymm0, ymm0, ymm0
        xor     eax, eax
1:      vmovaps [rsp + rax], ymm0
        vmovaps [rsp + rax + 32], ymm0
        add     rax, 64
        cmp     rax, r15
        jb      1b
        mov     rdx, r10                    // &W[k][j]
        xor     eax, eax                    // k*4
        mov     rcx, r9
        and     rcx, -16                    // K*4 rounded down to 4 rows
        jmp     4f
2:      vbroadcastss ymm0, dword ptr [r11 + rax]
        vbroadcastss ymm1, dword ptr [r11 + rax + 4]
        vbroadcastss ymm2, dword ptr [r11 + rax + 8]
        vbroadcastss ymm3, dword ptr [r11 + rax + 12]
        lea     rsi, [rdx + r8]
        lea     rdi, [rdx + r8*2]
        lea     r12, [rsi + r8*2]
        xor     ebx, ebx                    // strip offset
3:      vmovaps ymm4, [rsp + rbx]
        vmovaps ymm5, [rsp + rbx + 32]
        vfmadd231ps ymm4, ymm0, [rdx + rbx]
        vfmadd231ps ymm5, ymm0, [rdx + rbx + 32]
        vfmadd231ps ymm4, ymm1, [rsi + rbx]
        vfmadd231ps ymm5, ymm1, [rsi + rbx + 32]
        vfmadd231ps ymm4, ymm2, [rdi + rbx]
        vfmadd231ps ymm5, ymm2, [rdi + rbx + 32]
        vfmadd231ps ymm4, ymm3, [r12 + rbx]
        vfmadd231ps ymm5, ymm3, [r12 + rbx + 32]
        vmovaps [rsp + rbx], ymm4
        vmovaps [rsp + rbx + 32], ymm5
        add     rbx, 64
        cmp     rbx, r15
        jb      3b
        lea     rdx, [r12 + r8]             // next 4 rows
        add     rax, 16
4:      cmp     rax, rcx
        jb      2b
        jmp     7f
5:      vbroadcastss ymm0, dword ptr [r11 + rax]   // leftover rows (K % 4)
        xor     ebx, ebx
6:      vmovaps ymm4, [rsp + rbx]
        vfmadd231ps ymm4, ymm0, [rdx + rbx]
        vmovaps [rsp + rbx], ymm4
        add     rbx, 32
        cmp     rbx, r15
        jb      6b
        add     rdx, r8
        add     rax, 4
7:      cmp     rax, r9
        jb      5b
        mov     rbx, [rbp]                  // ctx (saved rbx)
        mov     rax, r14
        imul    rax, r8
        add     rax, [rbx + C_Y]
        lea     rax, [rax + r13*4]
        xor     ecx, ecx
8:      vmovaps ymm4, [rsp + rcx]
        vmovups [rax + rcx], ymm4
        add     rcx, 32
        cmp     rcx, r15
        jb      8b
        mov     rsp, rbp
        pop     rbx
        pop     rbp
        ret

// F32R R: R rows (1..6) x 16 columns, sharing each W load (ymm14/15).
.macro F32ST off, a, b
        lea     rax, [r14 + \off]
        imul    rax, r8
        add     rax, [rbx + C_Y]
        vmovups [rax + r13*4], \a
        vmovups [rax + r13*4 + 32], \b
.endm
.macro F32R R
f32r_\R:
        lea     r12, [r11 + r9]
        lea     r15, [r11 + r9*2]
        lea     rdi, [r12 + r9*2]
        lea     rsi, [r15 + r9*2]
        lea     rcx, [rdi + r9*2]
        vxorps  ymm0, ymm0, ymm0
        vxorps  ymm1, ymm1, ymm1
        vxorps  ymm2, ymm2, ymm2
        vxorps  ymm3, ymm3, ymm3
        vxorps  ymm4, ymm4, ymm4
        vxorps  ymm5, ymm5, ymm5
        vxorps  ymm6, ymm6, ymm6
        vxorps  ymm7, ymm7, ymm7
        vxorps  ymm8, ymm8, ymm8
        vxorps  ymm9, ymm9, ymm9
        vxorps  ymm10, ymm10, ymm10
        vxorps  ymm11, ymm11, ymm11
        mov     rdx, r10
        xor     eax, eax
        .p2align 5
1:      vmovups ymm14, [rdx]
        vmovups ymm15, [rdx + 32]
        vbroadcastss ymm12, dword ptr [r11 + rax]
        vfmadd231ps ymm0, ymm12, ymm14
        vfmadd231ps ymm1, ymm12, ymm15
.if \R >= 2
        vbroadcastss ymm13, dword ptr [r12 + rax]
        vfmadd231ps ymm2, ymm13, ymm14
        vfmadd231ps ymm3, ymm13, ymm15
.endif
.if \R >= 3
        vbroadcastss ymm12, dword ptr [r15 + rax]
        vfmadd231ps ymm4, ymm12, ymm14
        vfmadd231ps ymm5, ymm12, ymm15
.endif
.if \R >= 4
        vbroadcastss ymm13, dword ptr [rdi + rax]
        vfmadd231ps ymm6, ymm13, ymm14
        vfmadd231ps ymm7, ymm13, ymm15
.endif
.if \R >= 5
        vbroadcastss ymm12, dword ptr [rsi + rax]
        vfmadd231ps ymm8, ymm12, ymm14
        vfmadd231ps ymm9, ymm12, ymm15
.endif
.if \R >= 6
        vbroadcastss ymm13, dword ptr [rcx + rax]
        vfmadd231ps ymm10, ymm13, ymm14
        vfmadd231ps ymm11, ymm13, ymm15
.endif
        add     rdx, r8
        add     rax, 4
        cmp     rax, r9
        jb      1b
        F32ST   0, ymm0, ymm1
.if \R >= 2
        F32ST   1, ymm2, ymm3
.endif
.if \R >= 3
        F32ST   2, ymm4, ymm5
.endif
.if \R >= 4
        F32ST   3, ymm6, ymm7
.endif
.if \R >= 5
        F32ST   4, ymm8, ymm9
.endif
.if \R >= 6
        F32ST   5, ymm10, ymm11
.endif
        ret
.endm
        F32R 1
        F32R 2
        F32R 3
        F32R 4
        F32R 5
        F32R 6

// f32r_part: one row, the last partial unit (cnt = N - j in 1..15 columns);
// masked loads/stores so nothing outside the matrix / y is touched.
f32r_part:
        mov     rcx, [rbx + C_N]
        sub     rcx, r13                    // cnt
        mov     rax, rcx
        cmp     rax, 8
        jbe     1f
        mov     eax, 8
1:      sub     rcx, rax                    // c1 = cnt - c0
        lea     rdx, [rip + masktab + 32]
        neg     rax
        vmovdqu ymm14, [rdx + rax*4]        // first c0 lanes
        neg     rcx
        vmovdqu ymm15, [rdx + rcx*4]        // first c1 lanes
        vxorps  ymm0, ymm0, ymm0
        vxorps  ymm1, ymm1, ymm1
        mov     rdx, r10
        xor     eax, eax
2:      vbroadcastss ymm12, dword ptr [r11 + rax]
        vmaskmovps ymm2, ymm14, [rdx]
        vmaskmovps ymm3, ymm15, [rdx + 32]
        vfmadd231ps ymm0, ymm12, ymm2
        vfmadd231ps ymm1, ymm12, ymm3
        add     rdx, r8
        add     rax, 4
        cmp     rax, r9
        jb      2b
        mov     rax, r14
        imul    rax, r8
        add     rax, [rbx + C_Y]
        vmaskmovps [rax + r13*4], ymm14, ymm0
        vmaskmovps [rax + r13*4 + 32], ymm15, ymm1
        ret

// ===========================================================================
// f32r_chunk(ctx, u0, u1): pool_run worker for F32 (K,N): 16-column units
// [u0,u1), all M rows. M == 1 uses the 64-column kernel when 4 full units fit.
//   locals: [rsp] = u, [rsp+8] = u1, [rsp+16] = R
// ===========================================================================
f32r_chunk:
        CHUNK_ENTER
        mov     rbx, rcx
        mov     [rsp], rdx
        mov     [rsp + 8], r8
        mov     r9, [rbx + C_K]
        shl     r9, 2                       // K*4
        mov     r8, [rbx + C_N]
        shl     r8, 2                       // N*4
.Lf32c_u:
        mov     rax, [rsp]
        cmp     rax, [rsp + 8]
        jae     .Lf32c_done
        mov     r13, rax
        shl     r13, 4                      // j = 16u
        mov     r10, [rbx + C_W]
        lea     r10, [r10 + r13*4]
        xor     r14d, r14d
        lea     rax, [r13 + 16]
        cmp     rax, [rbx + C_N]
        ja      .Lf32c_part
        cmp     qword ptr [rbx + C_M], 1
        jne     .Lf32c_full
        mov     rax, [rsp + 8]
        sub     rax, [rsp]                  // units left in this chunk
        mov     rdx, [rbx + C_N]
        sub     rdx, r13
        shr     rdx, 4                      // full units left in the matrix
        cmp     rax, rdx
        cmova   rax, rdx
        cmp     rax, 16
        jbe     1f
        mov     eax, 16
1:      cmp     rax, 2
        jb      .Lf32c_full
        mov     [rsp + 16], rax
        mov     rcx, rax
        mov     r11, [rbx + C_X]
        call    f32_decw
        mov     rax, [rsp + 16]
        add     [rsp], rax
        jmp     .Lf32c_u
.Lf32c_full:
        mov     rax, [rbx + C_M]
        sub     rax, r14
        jle     .Lf32c_next
        mov     r11, r14
        imul    r11, r9
        add     r11, [rbx + C_X]
        cmp     rax, 6
        jbe     1f
        mov     eax, 6
1:      mov     [rsp + 16], rax
        lea     rcx, [rip + f32r_tab]
        call    [rcx + rax*8]
        add     r14, [rsp + 16]
        jmp     .Lf32c_full
.Lf32c_part:
        cmp     r14, [rbx + C_M]
        jae     .Lf32c_next
        mov     r11, r14
        imul    r11, r9
        add     r11, [rbx + C_X]
        call    f32r_part
        inc     r14
        jmp     .Lf32c_part
.Lf32c_next:
        inc     qword ptr [rsp]
        jmp     .Lf32c_u
.Lf32c_done:
        CHUNK_LEAVE

// ===========================================================================
// F32T (N,K) kernels: y[j] = dot(W[j,:], x). Per row: two 8-lane accumulators
// a0 (k%16 < 8) and a1 (k%16 >= 8) over full 16-blocks, then an 8-block into
// a0, then a masked remainder into a1, then (a0+a1) reduced hi128+lo128 ->
// movhl -> movshdup. f32t_k4 does 4 rows with exactly the per-row sequence of
// f32t_k1, so results never depend on the blocking.
//   in : r10 = &W[j][0], r11 = x, r8 = K*4, r13 = &out[0] (out[0..3] / out[0])
//   clobbers: rax, rcx, rdx, rdi, r12, r15, ymm0-ymm12
// ===========================================================================
.macro HSUM a, b                        // xmm\a[0] = sum of lanes of (ymm\a + ymm\b)
        vaddps  ymm\a, ymm\a, ymm\b
        vextractf128 xmm\b, ymm\a, 1
        vaddps  xmm\a, xmm\a, xmm\b
        vmovhlps xmm\b, xmm\a, xmm\a
        vaddps  xmm\a, xmm\a, xmm\b
        vmovshdup xmm\b, xmm\a
        vaddss  xmm\a, xmm\a, xmm\b
.endm
f32t_k4:
        lea     r12, [r10 + r8]
        lea     r15, [r10 + r8*2]
        lea     rdi, [r12 + r8*2]
        vxorps  ymm0, ymm0, ymm0
        vxorps  ymm1, ymm1, ymm1
        vxorps  ymm2, ymm2, ymm2
        vxorps  ymm3, ymm3, ymm3
        vxorps  ymm4, ymm4, ymm4
        vxorps  ymm5, ymm5, ymm5
        vxorps  ymm6, ymm6, ymm6
        vxorps  ymm7, ymm7, ymm7
        mov     rcx, r8
        and     rcx, -64
        xor     eax, eax
        jmp     2f
        .p2align 5
1:      vmovups ymm8, [r11 + rax]
        vmovups ymm9, [r11 + rax + 32]
        vfmadd231ps ymm0, ymm8, [r10 + rax]
        vfmadd231ps ymm1, ymm9, [r10 + rax + 32]
        vfmadd231ps ymm2, ymm8, [r12 + rax]
        vfmadd231ps ymm3, ymm9, [r12 + rax + 32]
        vfmadd231ps ymm4, ymm8, [r15 + rax]
        vfmadd231ps ymm5, ymm9, [r15 + rax + 32]
        vfmadd231ps ymm6, ymm8, [rdi + rax]
        vfmadd231ps ymm7, ymm9, [rdi + rax + 32]
        add     rax, 64
2:      cmp     rax, rcx
        jb      1b
        mov     rcx, r8
        sub     rcx, rax                    // remaining bytes 0..60
        cmp     rcx, 32
        jb      3f
        vmovups ymm8, [r11 + rax]
        vfmadd231ps ymm0, ymm8, [r10 + rax]
        vfmadd231ps ymm2, ymm8, [r12 + rax]
        vfmadd231ps ymm4, ymm8, [r15 + rax]
        vfmadd231ps ymm6, ymm8, [rdi + rax]
        add     rax, 32
        sub     rcx, 32
3:      test    rcx, rcx
        jz      4f
        lea     rdx, [rip + masktab + 32]
        sub     rdx, rcx
        vmovdqu ymm10, [rdx]
        vmaskmovps ymm8, ymm10, [r11 + rax]
        vmaskmovps ymm11, ymm10, [r10 + rax]
        vfmadd231ps ymm1, ymm8, ymm11
        vmaskmovps ymm11, ymm10, [r12 + rax]
        vfmadd231ps ymm3, ymm8, ymm11
        vmaskmovps ymm11, ymm10, [r15 + rax]
        vfmadd231ps ymm5, ymm8, ymm11
        vmaskmovps ymm11, ymm10, [rdi + rax]
        vfmadd231ps ymm7, ymm8, ymm11
4:      HSUM    0, 1
        HSUM    2, 3
        HSUM    4, 5
        HSUM    6, 7
        vmovss  dword ptr [r13], xmm0
        vmovss  dword ptr [r13 + 4], xmm2
        vmovss  dword ptr [r13 + 8], xmm4
        vmovss  dword ptr [r13 + 12], xmm6
        ret

f32t_k1:
        vxorps  ymm0, ymm0, ymm0
        vxorps  ymm1, ymm1, ymm1
        mov     rcx, r8
        and     rcx, -64
        xor     eax, eax
        jmp     2f
1:      vmovups ymm8, [r11 + rax]
        vmovups ymm9, [r11 + rax + 32]
        vfmadd231ps ymm0, ymm8, [r10 + rax]
        vfmadd231ps ymm1, ymm9, [r10 + rax + 32]
        add     rax, 64
2:      cmp     rax, rcx
        jb      1b
        mov     rcx, r8
        sub     rcx, rax
        cmp     rcx, 32
        jb      3f
        vmovups ymm8, [r11 + rax]
        vfmadd231ps ymm0, ymm8, [r10 + rax]
        add     rax, 32
        sub     rcx, 32
3:      test    rcx, rcx
        jz      4f
        lea     rdx, [rip + masktab + 32]
        sub     rdx, rcx
        vmovdqu ymm10, [rdx]
        vmaskmovps ymm8, ymm10, [r11 + rax]
        vmaskmovps ymm11, ymm10, [r10 + rax]
        vfmadd231ps ymm1, ymm8, ymm11
4:      HSUM    0, 1
        vmovss  dword ptr [r13], xmm0
        ret

// f32t_chunk(ctx, j0, j1): rows [j0,j1) of W (= output columns), all M rows of
// X; 4 W-rows at a time (they stay in L1 while the M activation rows pass).
//   locals: [rsp] = j, [rsp+8] = j1
f32t_chunk:
        CHUNK_ENTER
        mov     rbx, rcx
        mov     [rsp], rdx
        mov     [rsp + 8], r8
        mov     r8, [rbx + C_K]
        shl     r8, 2                       // K*4
.Lf32t_4:
        mov     rax, [rsp]
        lea     rcx, [rax + 4]
        cmp     rcx, [rsp + 8]
        ja      .Lf32t_1
        mov     r10, rax
        imul    r10, r8
        add     r10, [rbx + C_W]
        xor     r14d, r14d
1:      cmp     r14, [rbx + C_M]
        jae     2f
        mov     r11, r14
        imul    r11, r8
        add     r11, [rbx + C_X]
        mov     r13, r14
        imul    r13, [rbx + C_N]
        add     r13, [rsp]
        shl     r13, 2
        add     r13, [rbx + C_Y]
        call    f32t_k4
        inc     r14
        jmp     1b
2:      add     qword ptr [rsp], 4
        jmp     .Lf32t_4
.Lf32t_1:
        mov     rax, [rsp]
        cmp     rax, [rsp + 8]
        jae     .Lf32t_done
        mov     r10, rax
        imul    r10, r8
        add     r10, [rbx + C_W]
        xor     r14d, r14d
1:      cmp     r14, [rbx + C_M]
        jae     2f
        mov     r11, r14
        imul    r11, r8
        add     r11, [rbx + C_X]
        mov     r13, r14
        imul    r13, [rbx + C_N]
        add     r13, [rsp]
        shl     r13, 2
        add     r13, [rbx + C_Y]
        call    f32t_k1
        inc     r14
        jmp     1b
2:      inc     qword ptr [rsp]
        jmp     .Lf32t_1
.Lf32t_done:
        CHUNK_LEAVE

// ===========================================================================
// job_threads (internal): rcx = rows of the call -> rax = threads to use.
//   rows <= mv_mem_rows: the call is DRAM-bandwidth bound -> mv_mem_threads
//   (0 = pool_mem_threads = the P-cores; measured fastest, more threads only
//   add contention), else all pool threads (prefill is compute bound).
//   clobbers: rax, rcx, rdx
// ===========================================================================
job_threads:
        mov     rdx, rcx
        call    pool_threads
        cmp     rdx, [rip + mv_mem_rows]
        jg      1f
        mov     rcx, [rip + mv_mem_threads]
        test    rcx, rcx
        jnz     2f
        mov     rcx, [rip + pool_mem_threads]
2:      test    rcx, rcx
        jle     1f
        cmp     rcx, rax
        cmovb   rax, rcx
1:      ret

// ===========================================================================
// calc_grain (internal): rcx = units, rdx = work bytes per unit, r8 = threads
//   -> rax = grain. total < mv_small_bytes or 1 thread -> grain = units
//   (pool_run runs inline); else ceil(units / (threads*mv_chunks_per_thread)),
//   at least ceil(mv_min_chunk_bytes / bytes_per_unit), at least 1.
//   clobbers: rax, rcx, rdx, r8, r9, r10, r11
// ===========================================================================
calc_grain:
        mov     r10, rcx                    // units
        mov     r11, rdx                    // bytes per unit
        mov     rax, rcx
        imul    rax, rdx
        cmp     rax, [rip + mv_small_bytes]
        jb      .Lcg_all
        mov     rax, r8
        cmp     rax, 1
        jle     .Lcg_all
        imul    rax, [rip + mv_chunks_per_thread]
        mov     r8, rax                     // target chunks
        lea     rax, [r10 + r8 - 1]
        xor     edx, edx
        div     r8
        mov     r9, rax                     // grain by count
        mov     rax, [rip + mv_min_chunk_bytes]
        add     rax, r11
        dec     rax
        xor     edx, edx
        div     r11                         // grain by size
        cmp     rax, r9
        cmovb   rax, r9
        cmp     rax, 1
        jge     1f
        mov     eax, 1
1:      ret
.Lcg_all:
        mov     rax, r10
        ret

// ===========================================================================
// void matvec(const Mat* m, const float* x, float* y) = matmul_rows(m, x, 1, y)
// ===========================================================================
        .globl  matvec
matvec:
        mov     r9, r8
        mov     r8d, 1
        jmp     matmul_rows

// ===========================================================================
// void matmul_rows(const Mat* m, const float* X, i64 M, float* Y)
//   rcx = m, rdx = X (M x K floats), r8 = M, r9 = Y (M x N floats)
//   saves rbx rsi rdi r12-r15; uses only volatile ymm itself.
// ===========================================================================
        .globl  matmul_rows
matmul_rows:
        push    rbx
        push    rsi
        push    rdi
        push    r12
        push    r13
        push    r14
        push    r15
        sub     rsp, 64                     // shadow + arg5 [rsp+32] + locals
        mov     rbx, rcx
        mov     rsi, rdx
        mov     r12, r8
        mov     rdi, r9
        test    r12, r12
        jle     .Lmr_done
        lea     r15, [rip + ctx]
        mov     rax, [rbx + MAT_DATA]
        mov     [r15 + C_W], rax
        mov     rax, [rbx + MAT_K]
        mov     [r15 + C_K], rax
        mov     rax, [rbx + MAT_N]
        mov     [r15 + C_N], rax
        mov     rax, [rbx + MAT_KIND]
        cmp     rax, MATK_Q8
        je      .Lmr_q8
        mov     [r15 + C_X], rsi
        mov     [r15 + C_Y], rdi
        mov     [r15 + C_M], r12
        cmp     rax, MATK_F32
        je      .Lmr_f32
        cmp     rax, MATK_F32T
        je      .Lmr_f32t
        jmp     .Lmr_done

.Lmr_f32:                                   // units of 16 columns
        mov     rcx, [rbx + MAT_N]
        add     rcx, 15
        shr     rcx, 4
        mov     r13, rcx
        mov     rcx, r12
        call    job_threads
        mov     [rsp + 40], rax
        mov     rcx, r13
        mov     rdx, [rbx + MAT_K]
        shl     rdx, 6                      // 64 bytes of W per unit per k
        imul    rdx, r12
        mov     r8, [rsp + 40]
        call    calc_grain
        lea     rcx, [rip + f32r_chunk]
        mov     rdx, r15
        mov     r8, r13
        mov     r9, rax
        mov     rax, [rsp + 40]
        mov     [rsp + 32], rax
        call    pool_run_n
        jmp     .Lmr_done

.Lmr_f32t:                                  // units = rows of W
        mov     r13, [rbx + MAT_N]
        mov     rcx, r12
        call    job_threads
        mov     [rsp + 40], rax
        mov     rcx, r13
        mov     rdx, [rbx + MAT_K]
        shl     rdx, 2
        imul    rdx, r12
        mov     r8, [rsp + 40]
        call    calc_grain
        add     rax, 3
        and     rax, -4                     // keep 4-row blocks aligned
        lea     rcx, [rip + f32t_chunk]
        mov     rdx, r15
        mov     r8, r13
        mov     r9, rax
        mov     rax, [rsp + 40]
        mov     [rsp + 32], rax
        call    pool_run_n
        jmp     .Lmr_done

.Lmr_q8:
        cmp     qword ptr [rbx + MAT_PACKED], 0
        jne     1f
        mov     rcx, rbx
        call    mat_q8_pack
        test    rax, rax
        jz      .Lmr_done
1:      mov     rax, [rbx + MAT_PACKED]
        mov     [r15 + C_PACKED], rax
        mov     rax, [rbx + MAT_AUX]
        mov     [r15 + C_KPAD], rax
        mov     rax, [rbx + MAT_SCALE]
        mov     [r15 + C_SCALE], rax
        mov     rax, [rbx + MAT_COLSUM]
        mov     [r15 + C_COLSUM], rax
        lea     rax, [rip + xq_buf]
        mov     [r15 + C_XQ], rax
        lea     rax, [rip + as_buf]
        mov     [r15 + C_AS], rax
        mov     rax, [rip + cpu_features]
        shr     rax, 1
        and     eax, 1
        mov     [r15 + C_VNNI], rax
        shl     eax, 7                      // 0x80 for VNNI (u8 = q+128), 0 for AVX2
        mov     [r15 + C_XORB], rax
        // rows per block: min(M, MAXROWS, XQ_BYTES / Kpad)
        mov     eax, XQ_BYTES
        xor     edx, edx
        div     qword ptr [r15 + C_KPAD]
        cmp     rax, MAXROWS
        jbe     2f
        mov     eax, MAXROWS
2:      mov     [rsp + 48], rax             // max rows per block
        xor     r13d, r13d                  // m0
.Lmr_q8_blk:
        cmp     r13, r12
        jge     .Lmr_done
        mov     r14, r12
        sub     r14, r13
        cmp     r14, [rsp + 48]
        jbe     3f
        mov     r14, [rsp + 48]             // r14 = Mc
3:      mov     [r15 + C_M], r14
        mov     rax, r13
        imul    rax, [rbx + MAT_K]
        lea     rax, [rsi + rax*4]
        mov     [r15 + C_X], rax
        mov     rax, r13
        imul    rax, [rbx + MAT_N]
        lea     rax, [rdi + rax*4]
        mov     [r15 + C_Y], rax
        // quantize the block's activation rows
        cmp     r14, 1
        jne     4f
        mov     rcx, [r15 + C_X]
        mov     rdx, [rbx + MAT_K]
        lea     r8, [rip + xq_buf]
        mov     r9, [r15 + C_XORB]
        call    quant_row
        vmovss  dword ptr [rip + as_buf], xmm0
        vzeroupper
        jmp     5f
4:      call    pool_threads
        lea     r8, [rax + rax]             // ~2 chunks per thread
        lea     rax, [r14 + r8 - 1]
        xor     edx, edx
        div     r8
        lea     rcx, [rip + qrows_chunk]
        mov     rdx, r15
        mov     r8, r14
        mov     r9, rax
        call    pool_run
5:      // the matmul itself: units = panels of 16 columns
        mov     rcx, [rbx + MAT_N]
        add     rcx, 15
        shr     rcx, 4
        mov     [rsp + 56], rcx
        mov     rcx, r14
        call    job_threads
        mov     [rsp + 40], rax
        mov     rcx, [rsp + 56]
        mov     rdx, [r15 + C_KPAD]
        shl     rdx, 4
        imul    rdx, r14
        mov     r8, [rsp + 40]
        call    calc_grain
        lea     rcx, [rip + q8_chunk]
        mov     rdx, r15
        mov     r8, [rsp + 56]
        mov     r9, rax
        mov     rax, [rsp + 40]
        mov     [rsp + 32], rax
        call    pool_run_n
        add     r13, r14
        jmp     .Lmr_q8_blk

.Lmr_done:
        vzeroupper
        add     rsp, 64
        pop     r15
        pop     r14
        pop     r13
        pop     r12
        pop     rdi
        pop     rsi
        pop     rbx
        ret

// ===========================================================================
// void* mat_q8_pack(Mat* m)
//   rcx = m (kind Q8, K, N, data = int8 (K,N) row-major, scale set)
//   allocates Kpad*Npad (packed) + Npad*4 (colsum) bytes with VirtualAlloc,
//   fills them, sets m->packed / m->colsum / m->aux = Kpad.
//   returns rax = packed (0 on allocation failure). Saves rbx rsi rdi r12-r15.
// ===========================================================================
        .globl  mat_q8_pack
mat_q8_pack:
        push    rbx
        push    rsi
        push    rdi
        push    r12
        push    r13
        push    r14
        push    r15
        sub     rsp, 48
        mov     rbx, rcx
        mov     r12, [rbx + MAT_K]
        mov     r13, [rbx + MAT_N]
        lea     r14, [r12 + 15]
        and     r14, -16                    // Kpad
        lea     r15, [r13 + 15]
        and     r15, -16                    // Npad
        mov     rdx, r14
        imul    rdx, r15
        lea     rdx, [rdx + r15*4]          // bytes
        xor     ecx, ecx
        mov     r8d, 0x3000                 // MEM_COMMIT | MEM_RESERVE
        mov     r9d, 4                      // PAGE_READWRITE
        call    VirtualAlloc
        test    rax, rax
        jz      .Lpk_ret
        mov     rdi, rax                    // out cursor
        mov     rsi, r14
        imul    rsi, r15
        add     rsi, rax                    // colsum (zeroed by VirtualAlloc)
        mov     [rbx + MAT_COLSUM], rsi
        mov     [rbx + MAT_AUX], r14
        mov     r10, [rbx + MAT_DATA]
        // ---- pack: panel p (16 cols) -> kgroup g -> half h ----
        xor     r8d, r8d                    // j0 of the panel
.Lpk_p:
        cmp     r8, r15
        jae     .Lpk_cs
        xor     r9d, r9d                    // k
.Lpk_g:
        cmp     r9, r14
        jae     .Lpk_pnext
        mov     rax, r8
        call    pack8
        add     rdi, 32
        lea     rax, [r8 + 8]
        call    pack8
        add     rdi, 32
        add     r9, 4
        jmp     .Lpk_g
.Lpk_pnext:
        add     r8, 16
        jmp     .Lpk_p
        // ---- colsum[j] = sum_k W[k][j], then *128 ----
.Lpk_cs:
        xor     r8d, r8d                    // k
        mov     r11, r13
        and     r11, -8
.Lpk_ck:
        cmp     r8, r12
        jae     .Lpk_shift
        mov     rcx, r8
        imul    rcx, r13
        add     rcx, r10                    // row k
        xor     eax, eax
        jmp     2f
1:      vpmovsxbd ymm0, qword ptr [rcx + rax]
        vpaddd  ymm0, ymm0, [rsi + rax*4]
        vmovdqu [rsi + rax*4], ymm0
        add     rax, 8
2:      cmp     rax, r11
        jb      1b
        jmp     4f
3:      movsx   edx, byte ptr [rcx + rax]
        add     [rsi + rax*4], edx
        inc     rax
4:      cmp     rax, r13
        jb      3b
        inc     r8
        jmp     .Lpk_ck
.Lpk_shift:
        xor     eax, eax
1:      vmovdqu ymm0, [rsi + rax*4]
        vpslld  ymm0, ymm0, 7
        vmovdqu [rsi + rax*4], ymm0
        add     rax, 8
        cmp     rax, r15
        jb      1b
        vzeroupper
        mov     rax, r14
        imul    rax, r15
        neg     rax
        add     rax, rsi                    // packed base
        mov     [rbx + MAT_PACKED], rax
.Lpk_ret:
        add     rsp, 48
        pop     r15
        pop     r14
        pop     r13
        pop     r12
        pop     rdi
        pop     rsi
        pop     rbx
        ret

// pack8 (internal): write the 32-byte block for columns rax..rax+7, k = r9..r9+3
//   in : rax = j, r9 = k, rdi = out, r10 = data, r12 = K, r13 = N
//   clobbers rcx, rdx, r11, xmm0-xmm3
pack8:
        lea     rcx, [r9 + 4]
        cmp     rcx, r12
        ja      .Lp8_slow
        lea     rcx, [rax + 8]
        cmp     rcx, r13
        ja      .Lp8_slow
        mov     rcx, r9
        imul    rcx, r13
        add     rcx, r10
        add     rcx, rax                    // &W[k][j]
        vmovq   xmm0, qword ptr [rcx]
        vmovq   xmm1, qword ptr [rcx + r13]
        lea     rdx, [rcx + r13*2]
        vmovq   xmm2, qword ptr [rdx]
        vmovq   xmm3, qword ptr [rdx + r13]
        vpunpcklbw xmm0, xmm0, xmm1         // (r0,r1) byte pairs per column
        vpunpcklbw xmm2, xmm2, xmm3         // (r2,r3)
        vpunpcklwd xmm1, xmm0, xmm2         // cols 0-3: r0 r1 r2 r3
        vpunpckhwd xmm3, xmm0, xmm2         // cols 4-7
        vmovdqu [rdi], xmm1
        vmovdqu [rdi + 16], xmm3
        ret
.Lp8_slow:                                  // edge block: bounds-checked bytes
        push    rbx
        push    rsi
        xor     ecx, ecx                    // c
1:      xor     edx, edx                    // t
2:      xor     r11d, r11d
        lea     rbx, [r9 + rdx]
        cmp     rbx, r12
        jae     3f
        lea     rsi, [rax + rcx]
        cmp     rsi, r13
        jae     3f
        imul    rbx, r13
        add     rbx, rsi
        movzx   r11d, byte ptr [r10 + rbx]
3:      lea     rbx, [rdi + rcx*4]
        mov     [rbx + rdx], r11b
        inc     rdx
        cmp     rdx, 4
        jb      2b
        inc     rcx
        cmp     rcx, 8
        jb      1b
        pop     rsi
        pop     rbx
        ret

// ===========================================================================
// void* mat_q8_from_f32(Mat* out, const float* w, i64 K, i64 N)
//   = quantize_int8_percol(w, K, N, data, scale) (bit-identical) + mat_q8_pack.
//   One VirtualAlloc holds data (K*N, rounded to 64) followed by scale (N floats).
//   8 columns at a time: pass 1 max|w| down the columns, pass 2 quantize.
//   returns rax = packed (0 on failure). Saves rbx rsi rdi r12-r15.
// ===========================================================================
        .globl  mat_q8_from_f32
mat_q8_from_f32:
        push    rbx
        push    rsi
        push    rdi
        push    r12
        push    r13
        push    r14
        push    r15
        sub     rsp, 48
        mov     rbx, rcx
        mov     rsi, rdx
        mov     r12, r8                     // K
        mov     r13, r9                     // N
        mov     r14, r12
        imul    r14, r13
        add     r14, 63
        and     r14, -64                    // data bytes
        lea     rdx, [r14 + r13*4]
        xor     ecx, ecx
        mov     r8d, 0x3000
        mov     r9d, 4
        call    VirtualAlloc
        test    rax, rax
        jz      .Lqf_ret
        mov     rdi, rax                    // data
        add     r14, rax                    // scale
        mov     qword ptr [rbx + MAT_KIND], MATK_Q8
        mov     [rbx + MAT_K], r12
        mov     [rbx + MAT_N], r13
        mov     [rbx + MAT_DATA], rdi
        mov     [rbx + MAT_SCALE], r14
        mov     qword ptr [rbx + MAT_PACKED], 0
        mov     qword ptr [rbx + MAT_COLSUM], 0
        mov     qword ptr [rbx + MAT_AUX], 0
        lea     r15, [r13*4]                // row stride in bytes
        vbroadcastss ymm5, dword ptr [rip + c_absmask]
        xor     r8d, r8d                    // j
.Lqf_j:
        lea     rax, [r8 + 8]
        cmp     rax, r13
        ja      .Lqf_tail
        vxorps  ymm0, ymm0, ymm0            // pass 1: column max
        lea     rcx, [rsi + r8*4]
        mov     rdx, r12
1:      vandps  ymm1, ymm5, [rcx]
        vmaxps  ymm0, ymm0, ymm1
        add     rcx, r15
        dec     rdx
        jnz     1b
        vdivps  ymm1, ymm0, [rip + v_127f]  // s = mx/127
        vxorps  ymm2, ymm2, ymm2
        vcmpps  ymm2, ymm0, ymm2, 0x1e      // mx > 0 (GT_OQ)
        vmovups ymm3, [rip + v_1f]
        vblendvps ymm1, ymm3, ymm1, ymm2    // mx > 0 ? s : 1
        vmovups [r14 + r8*4], ymm1
        vdivps  ymm3, ymm3, ymm1            // inv = 1/s
        lea     rcx, [rsi + r8*4]           // pass 2: quantize
        lea     r9, [rdi + r8]
        mov     rdx, r12
2:      vmulps  ymm0, ymm3, [rcx]
        vcvtps2dq ymm0, ymm0
        vpminsd ymm0, ymm0, [rip + v_127i]
        vpmaxsd ymm0, ymm0, [rip + v_m127i]
        vextracti128 xmm1, ymm0, 1
        vpackssdw xmm0, xmm0, xmm1
        vpacksswb xmm0, xmm0, xmm0
        vmovq   qword ptr [r9], xmm0
        add     rcx, r15
        add     r9, r13
        dec     rdx
        jnz     2b
        add     r8, 8
        jmp     .Lqf_j
.Lqf_tail:                                  // remaining columns one by one
        cmp     r8, r13
        jae     .Lqf_pack
        vxorps  xmm0, xmm0, xmm0
        lea     rcx, [rsi + r8*4]
        mov     rdx, r12
3:      vmovss  xmm1, dword ptr [rcx]
        vandps  xmm1, xmm1, xmm5
        vmaxss  xmm0, xmm0, xmm1
        add     rcx, r15
        dec     rdx
        jnz     3b
        vmovss  xmm3, dword ptr [rip + c_one]
        vxorps  xmm2, xmm2, xmm2
        vucomiss xmm0, xmm2
        jbe     4f
        vdivss  xmm3, xmm0, dword ptr [rip + c_127f]
4:      vmovss  dword ptr [r14 + r8*4], xmm3
        vmovss  xmm4, dword ptr [rip + c_one]
        vdivss  xmm4, xmm4, xmm3
        lea     rcx, [rsi + r8*4]
        lea     r9, [rdi + r8]
        mov     rdx, r12
5:      vmulss  xmm0, xmm4, dword ptr [rcx]
        vcvtss2si eax, xmm0
        cmp     eax, 127
        jle     6f
        mov     eax, 127
6:      cmp     eax, -127
        jge     7f
        mov     eax, -127
7:      mov     [r9], al
        add     rcx, r15
        add     r9, r13
        dec     rdx
        jnz     5b
        inc     r8
        jmp     .Lqf_tail
.Lqf_pack:
        vzeroupper
        mov     rcx, rbx
        call    mat_q8_pack
.Lqf_ret:
        add     rsp, 48
        pop     r15
        pop     r14
        pop     r13
        pop     r12
        pop     rdi
        pop     rsi
        pop     rbx
        ret
