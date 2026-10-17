// ops.S - elementwise / per-row operators of the decoder (owner: decode agent)
//
// Everything here is AVX2 + FMA, Win64 ABI, no CRT. The functions mirror the C
// reference engine (../rmsnorm.c, ../rope.c, ../llama.c decode_attention,
// ../swiglu.c) operation by operation, so the asm port computes the same numbers
// (bit-exact where the C order of operations can be kept, within a few ulp where
// a vector reduction must reorder a sum).
//
// Exported functions (all ints are int64, floats are f32):
//   void rmsnorm(const float* x, const float* g, float* out, i64 n, float eps)
//          out = x * r * g,  r = 1/sqrtf(mean(x^2) + eps)        (out may == x)
//   void rmsnorm_rows(const float* x, const float* g, float* out, i64 n, float eps, i64 rows)
//          rmsnorm of `rows` consecutive rows of n floats
//   void rmsnorm_heads(float* x, const float* g, i64 n_heads, i64 head_dim, float eps)
//          in-place per-head RMSNorm (Qwen3 QK-norm): the gain g has head_dim entries
//   void vec_add(float* y, const float* x, i64 n)             y += x (residual)
//   void add_rows(float* y, const float* b, i64 n, i64 rows)  y[r][j] += b[j] (bias)
//   void vec_copy(float* dst, const float* src, i64 n)
//   void silu_mul(float* g, const float* u, i64 n)            g = SiLU(g) * u
//   void expf_vec(float* dst, const float* src, i64 n)        dst = expf(src), bit-exact
//          with the expf the C engine links (zig's compiler_rt = musl/FreeBSD e_expf.c)
//   void rope_inv_freq(float* inv, i64 head_dim, float base)  inv[k] = powf(base, -2k/hd)
//   void rope_table(float* cs, const float* inv, i64 half, i64 pos)
//          cs[k] = cosf(pos*inv[k]), cs[half+k] = sinf(pos*inv[k])
//   void rope_apply(float* v, i64 n_heads, i64 head_dim, const float* cs)
//   void rope_apply_rows(float* v, i64 n_heads, i64 head_dim, const float* cs, i64 rows)
//          row r uses table cs + r*head_dim (one table per position)
//   void rope_pos(float* v, i64 n_heads, i64 head_dim, float base, i64 pos)
//          = rope_heads_pos of ../rope.c for ONE row (convenience; decode uses the
//            table functions so the transcendental work is done once per token)
//   void kv_put(float* kcache, i64 cap, i64 slot, const float* row, i64 kv_dim)
//          store a K row into the cache (blocked layout, see "KV cache layout")
//   void attention(AttnArgs* a)   single-query GQA attention over the KV ring cache,
//          for nrows consecutive query rows (decode: 1 row; prefill: a batch),
//          threaded over (row, head) tasks with pool_run when the work is big.
//
// Win64 ABI reminders: args rcx, rdx, r8, r9, then [rsp+40], [rsp+48]... at entry;
// float args travel in xmm<position>; callee-saved: rbx rbp rdi rsi r12-r15 xmm6-15.
// Every function that touches ymm executes vzeroupper before returning.

#include "model.inc"
        .intel_syntax noprefix

// ---------------------------------------------------------------------------
// AttnArgs: the parameter block of attention() (shared with decode.S, which
// keeps an identical copy of these #defines).
// ---------------------------------------------------------------------------
#define AT_Q        0     // float* q of row 0, rows of n_heads*hd floats
#define AT_KC       8     // float* this layer's K cache (cap slots x kv_dim)
#define AT_VC       16    // float* V cache
#define AT_CAP      24    // ring slots of this layer (position p lives in slot p % cap)
#define AT_WINDOW   32    // sliding window (<= 0: full causal)
#define AT_POS0     40    // absolute position of row 0
#define AT_NROWS    48    // number of query rows (row i is at position pos0 + i)
#define AT_NHEADS   56
#define AT_NKV      64
#define AT_HD       72    // head_dim (multiple of 8)
#define AT_SC       80    // float* score scratch: one block of AT_SCSTRIDE floats per task
#define AT_SCSTRIDE 88    // floats per task, >= roundup8(longest key span + 7) + 8
#define AT_O        96    // float* output rows (n_heads*hd floats each)
#define AT_SCALE    104   // f32, filled in by attention(): 1/sqrtf(hd)
#define AT_GROUP    112   // i64, filled in by attention(): n_heads / n_kv

// Below this much work (tasks x keys x head_dim) attention runs on the calling
// thread: the pool hand-off costs a few microseconds.
#define ATTN_THREAD_MIN (1 << 17)

// ---------------------------------------------------------------------------
// Constants. A vector constant is stored 8 times so it can be used directly as a
// 256-bit memory operand (vmulps ymm, ymm, [const]) without a broadcast.
// ---------------------------------------------------------------------------
        .section .rdata,"dr"
        .p2align 5
.macro CONST8I name, val
\name:
        .rept 8
        .long \val
        .endr
.endm
.macro CONST8F name, val
\name:
        .rept 8
        .float \val
        .endr
.endm

        CONST8I C_ABS,    0x7fffffff
        CONST8I C_SIGN,   0x80000000
        CONST8F C_HALF,   0.5
        CONST8F C_ONE,    1.0
        CONST8F C_TWO,    2.0
        // musl / FreeBSD e_expf.c constants (the assembler rounds the decimal
        // literals to f32 exactly like the C/zig compiler does)
        CONST8F C_INVLN2, 1.4426950216e+0
        CONST8F C_LN2HI,  6.9314575195e-1
        CONST8F C_LN2LO,  1.4286067653e-6
        CONST8F C_P1,     1.6666625440e-1
        CONST8F C_P2,     -2.7667332906e-3
        CONST8I C_TH1,    0x3EB17218    // |x| > 0.5*ln2 : needs range reduction
        CONST8I C_TH2,    0x3F851592    // |x| > 1.5*ln2 : k from rounding, else k = +-1
        CONST8I C_B127,   127           // float exponent bias
        CONST8I C_SMALL,  0x39000000    // |x| <= 2^-13 : expf(x) = 1 + x
        CONST8I C_OVF,    0x42b17218    // x >= 88.722839 : +inf
        CONST8I C_UNF,    0xC2CFF1B5    // x <= -103.972084 : 0
        CONST8I C_INF,    0x7f800000
        CONST8I C_NEGINF, 0xff800000
C_IOTA: .long 0, 1, 2, 3, 4, 5, 6, 7
C_MTWO: .float -2.0

        .text

// ---------------------------------------------------------------------------
// EXP8 x, t1..t6 : x = expf(x) for 8 lanes, BIT-EXACT with the scalar expf the C
// engine links (zig compiler_rt `expf`, a port of musl's / FreeBSD's e_expf.c):
//
//   k  = 0                          if |x| <= 0.5 ln2
//        +-1                        if |x| <= 1.5 ln2
//        (int)(invln2*x +- 0.5)     otherwise (truncation toward zero)
//   hi = x - k*ln2hi ; lo = k*ln2lo ; r = hi - lo          (ln2 split in 2 floats)
//   c  = r - r^2*(P1 + r^2*P2)                              (degree-4 minimax)
//   y  = 1 + (r*c/(2-c) - lo + hi)                          (Remez rational form)
//   expf(x) = y * 2^k
// plus the special cases: |x| <= 2^-13 -> 1+x, x >= 88.72 -> inf, x <= -103.97 -> 0,
// NaN -> NaN. Every scalar operation of the C code is one vector operation here
// with the same operands in the same order (no FMA contraction: the reference is
// compiled without it), so each lane rounds exactly like the scalar code.
// The branches become blends. 2^k is applied as two exact power-of-two factors
// 2^(k>>1) * 2^(k-(k>>1)) so that k in [-150, 128] never leaves the normal range
// of the factor itself; the final multiply rounds once, exactly like scalbnf.
// ---------------------------------------------------------------------------
.macro EXP8 x, t1, t2, t3, t4, t5, t6
        vandps    \t1, \x, ymmword ptr [rip + C_ABS]     // t1 = |x| bits ("hx")
        vandps    \t2, \x, ymmword ptr [rip + C_SIGN]    // t2 = sign bit of x
        vorps     \t3, \t2, ymmword ptr [rip + C_HALF]   // t3 = half[sign] = +-0.5
        vmulps    \t4, \x, ymmword ptr [rip + C_INVLN2]  // invln2 * x
        vaddps    \t4, \t4, \t3                          //   + half[sign]
        vroundps  \t4, \t4, 3                            // trunc -> k (as float), case 3
        vorps     \t3, \t2, ymmword ptr [rip + C_ONE]    // +-1.0 = k for case 2
        vpcmpgtd  \t5, \t1, ymmword ptr [rip + C_TH1]    // case 2 or 3 ?
        vandps    \t3, \t3, \t5                          // k = 0 in case 1
        vpcmpgtd  \t5, \t1, ymmword ptr [rip + C_TH2]    // case 3 ?
        vblendvps \t3, \t3, \t4, \t5                     // t3 = fk (float k)
        vmulps    \t4, \t3, ymmword ptr [rip + C_LN2HI]  // fk * ln2hi
        vsubps    \t4, \x, \t4                           // hi = x - fk*ln2hi
        vmulps    \t5, \t3, ymmword ptr [rip + C_LN2LO]  // lo = fk*ln2lo
        vsubps    \t2, \t4, \t5                          // r = hi - lo
        vcvtps2dq \t3, \t3                               // k as int32 (exact)
        vmulps    \t1, \t2, \t2                          // xx = r*r
        vmulps    \t6, \t1, ymmword ptr [rip + C_P2]     // xx*P2
        vaddps    \t6, \t6, ymmword ptr [rip + C_P1]     // P1 + xx*P2
        vmulps    \t6, \t6, \t1                          // xx*(P1 + xx*P2)
        vsubps    \t6, \t2, \t6                          // c = r - xx*(...)
        vmulps    \t1, \t2, \t6                          // r*c
        vmovaps   \t2, ymmword ptr [rip + C_TWO]
        vsubps    \t2, \t2, \t6                          // 2 - c
        vdivps    \t1, \t1, \t2                          // r*c/(2-c)
        vsubps    \t1, \t1, \t5                          //   - lo
        vaddps    \t1, \t1, \t4                          //   + hi
        vaddps    \t1, \t1, ymmword ptr [rip + C_ONE]    // y = 1 + (...)
        vpsrad    \t4, \t3, 1                            // k1 = k >> 1
        vpsubd    \t3, \t3, \t4                          // k2 = k - k1
        vpaddd    \t4, \t4, ymmword ptr [rip + C_B127]
        vpslld    \t4, \t4, 23                           // 2^k1 as a float bit pattern
        vpaddd    \t3, \t3, ymmword ptr [rip + C_B127]
        vpslld    \t3, \t3, 23                           // 2^k2
        vmulps    \t1, \t1, \t4                          // exact
        vmulps    \t1, \t1, \t3                          // y*2^k, one rounding
        // special cases (blends)
        vandps    \t2, \x, ymmword ptr [rip + C_ABS]     // |x|
        vcmpps    \t3, \t2, ymmword ptr [rip + C_SMALL], 2   // |x| <= 2^-13
        vaddps    \t4, \x, ymmword ptr [rip + C_ONE]     // 1 + x
        vblendvps \t1, \t1, \t4, \t3
        vcmpps    \t3, \x, ymmword ptr [rip + C_OVF], 13     // x >= 88.722839
        vblendvps \t1, \t1, ymmword ptr [rip + C_INF], \t3
        vcmpps    \t3, \x, ymmword ptr [rip + C_UNF], 2      // x <= -103.972084
        vandnps   \t1, \t3, \t1                          // -> 0
        vcmpps    \t3, \x, \x, 3                         // NaN -> NaN
        vblendvps \x, \t1, \x, \t3
.endm

// HSUM y, x, ty, tx : horizontal sum of the 8 lanes of ymm y into the low lane of
// its xmm alias x (tx = xmm alias of the temp ty).
.macro HSUM y, x, ty, tx
        vextractf128 \tx, \y, 1
        vaddps    \x, \x, \tx
        vmovhlps  \tx, \tx, \x
        vaddps    \x, \x, \tx
        vmovshdup \tx, \x
        vaddss    \x, \x, \tx
.endm

// ===========================================================================
// RMSNorm
// ===========================================================================

// rms_core (internal): one row.  in: rcx = x, rdx = g, r8 = out, r9 = n,
// xmm5 = eps.  Clobbers rax, r10, ymm0-ymm4.  Does NOT modify rcx/rdx/r8/r9/r11
// or xmm5, so callers can loop over rows cheaply. No stack use.
//
// C (rmsnorm.c rms_fwd_rows):  ms = sum x^2 ; ms *= 1.0f/n ; r = 1.0f/sqrtf(ms+eps);
//                              y = x * r * g
// The sum of squares is ONE serial chain ms = fma(x[j], x[j], ms), j ascending:
// that is exactly what clang emits for `ms += x[j]*x[j]` (FMA contraction on, no
// reassociation), so ms -- and everything after it -- is bit-identical to C.
// (A 4-accumulator vector sum would be faster and more accurate, but the int8
// path re-quantizes every activation, so a last-bit difference here can flip a
// rounding and make greedy decoding pick a different token than the C engine.)
// Cost: n * 4 cycles of FMA latency, ~1 us for n = 896 -- negligible next to a matvec.
        .p2align 4
rms_core:
        vxorps  xmm0, xmm0, xmm0                // ms = 0
        xor     eax, eax                        // j = 0
        mov     r10, r9
        and     r10, -4
        jz      2f
1:      vmovss  xmm4, dword ptr [rcx + rax*4]           // 4 per iteration (still
        vfmadd231ss xmm0, xmm4, xmm4                    // one dependency chain)
        vmovss  xmm4, dword ptr [rcx + rax*4 + 4]
        vfmadd231ss xmm0, xmm4, xmm4
        vmovss  xmm4, dword ptr [rcx + rax*4 + 8]
        vfmadd231ss xmm0, xmm4, xmm4
        vmovss  xmm4, dword ptr [rcx + rax*4 + 12]
        vfmadd231ss xmm0, xmm4, xmm4
        add     rax, 4
        cmp     rax, r10
        jb      1b
2:      cmp     rax, r9
        jae     6f
        vmovss  xmm4, dword ptr [rcx + rax*4]
        vfmadd231ss xmm0, xmm4, xmm4
        inc     rax
        jmp     2b
6:      vcvtsi2ss xmm1, xmm1, r9                        // (float)n
        vmovss  xmm2, dword ptr [rip + C_ONE]
        vdivss  xmm2, xmm2, xmm1                        // inv_n = 1.0f / n
        vmulss  xmm0, xmm0, xmm2                        // ms *= inv_n
        vaddss  xmm0, xmm0, xmm5                        // ms + eps
        vsqrtss xmm0, xmm0, xmm0                        // sqrtf (correctly rounded)
        vmovss  xmm1, dword ptr [rip + C_ONE]
        vdivss  xmm0, xmm1, xmm0                        // r = 1.0f / sqrtf(...)
        vbroadcastss ymm0, xmm0
        xor     eax, eax
        mov     r10, r9
        and     r10, -8
        jz      8f
7:      vmulps  ymm1, ymm0, ymmword ptr [rcx + rax*4]   // x * r
        vmulps  ymm1, ymm1, ymmword ptr [rdx + rax*4]   // (x * r) * g
        vmovups ymmword ptr [r8 + rax*4], ymm1
        add     rax, 8
        cmp     rax, r10
        jb      7b
8:      cmp     rax, r9
        jae     10f
9:      vmulss  xmm1, xmm0, dword ptr [rcx + rax*4]
        vmulss  xmm1, xmm1, dword ptr [rdx + rax*4]
        vmovss  dword ptr [r8 + rax*4], xmm1
        inc     rax
        cmp     rax, r9
        jb      9b
10:     ret

// void rmsnorm(const float* x, const float* g, float* out, i64 n, float eps)
// eps is the 5th argument: on the stack at [rsp+40] (above the return address
// and the caller's 32-byte shadow space).
        .globl  rmsnorm
        .p2align 4
rmsnorm:
        vmovss  xmm5, dword ptr [rsp + 40]
        call    rms_core
        vzeroupper
        ret

// void rmsnorm_rows(x, g, out, i64 n, float eps, i64 rows)
        .globl  rmsnorm_rows
        .p2align 4
rmsnorm_rows:
        vmovss  xmm5, dword ptr [rsp + 40]
        mov     r11, qword ptr [rsp + 48]       // rows
        test    r11, r11
        jle     2f
1:      call    rms_core
        lea     rcx, [rcx + r9*4]               // next input row
        lea     r8, [r8 + r9*4]                 // next output row
        dec     r11
        jnz     1b
2:      vzeroupper
        ret

// void rmsnorm_heads(float* x, const float* g, i64 n_heads, i64 head_dim, float eps)
// In place: every head_dim-long slice of x is one RMSNorm row (rmsnorm_heads in
// rmsnorm.c is exactly that: a (n_heads, head_dim) view of the vector).
        .globl  rmsnorm_heads
        .p2align 4
rmsnorm_heads:
        vmovss  xmm5, dword ptr [rsp + 40]
        mov     r11, r8                         // head count
        test    r11, r11
        jle     2f
1:      mov     r8, rcx                         // out = x (in place)
        call    rms_core
        lea     rcx, [rcx + r9*4]
        dec     r11
        jnz     1b
2:      vzeroupper
        ret

// ===========================================================================
// Small vector helpers
// ===========================================================================

// void vec_add(float* y, const float* x, i64 n) : y[i] += x[i]
// (C: x.data[j] += ao.data[j], bias_add -- a single rounded add per element, so
//  bit-exact in any order.)
        .globl  vec_add
        .p2align 4
vec_add:
        xor     eax, eax
        mov     r9, r8
        and     r9, -32
        jz      2f
1:      vmovups ymm0, ymmword ptr [rcx + rax*4]
        vmovups ymm1, ymmword ptr [rcx + rax*4 + 32]
        vmovups ymm2, ymmword ptr [rcx + rax*4 + 64]
        vmovups ymm3, ymmword ptr [rcx + rax*4 + 96]
        vaddps  ymm0, ymm0, ymmword ptr [rdx + rax*4]
        vaddps  ymm1, ymm1, ymmword ptr [rdx + rax*4 + 32]
        vaddps  ymm2, ymm2, ymmword ptr [rdx + rax*4 + 64]
        vaddps  ymm3, ymm3, ymmword ptr [rdx + rax*4 + 96]
        vmovups ymmword ptr [rcx + rax*4], ymm0
        vmovups ymmword ptr [rcx + rax*4 + 32], ymm1
        vmovups ymmword ptr [rcx + rax*4 + 64], ymm2
        vmovups ymmword ptr [rcx + rax*4 + 96], ymm3
        add     rax, 32
        cmp     rax, r9
        jb      1b
2:      mov     r9, r8
        and     r9, -8
        cmp     rax, r9
        jae     4f
3:      vmovups ymm0, ymmword ptr [rcx + rax*4]
        vaddps  ymm0, ymm0, ymmword ptr [rdx + rax*4]
        vmovups ymmword ptr [rcx + rax*4], ymm0
        add     rax, 8
        cmp     rax, r9
        jb      3b
4:      cmp     rax, r8
        jae     6f
5:      vmovss  xmm0, dword ptr [rcx + rax*4]
        vaddss  xmm0, xmm0, dword ptr [rdx + rax*4]
        vmovss  dword ptr [rcx + rax*4], xmm0
        inc     rax
        cmp     rax, r8
        jb      5b
6:      vzeroupper
        ret

// void add_rows(float* y, const float* b, i64 n, i64 rows) : y[r*n + j] += b[j]
// (the bias of a batch of rows). Calls vec_add per row; vec_add only touches
// rax, r9 and ymm0-3, so rcx/rdx/r8/r10/r11 survive the call.
        .globl  add_rows
        .p2align 4
add_rows:
        mov     r11, r9
        test    r11, r11
        jle     2f
1:      mov     r10, rcx
        call    vec_add
        lea     rcx, [r10 + r8*4]
        dec     r11
        jnz     1b
2:      ret

// void vec_copy(float* dst, const float* src, i64 n)
        .globl  vec_copy
        .p2align 4
vec_copy:
        xor     eax, eax
        mov     r9, r8
        and     r9, -8
        jz      2f
1:      vmovups ymm0, ymmword ptr [rdx + rax*4]
        vmovups ymmword ptr [rcx + rax*4], ymm0
        add     rax, 8
        cmp     rax, r9
        jb      1b
2:      cmp     rax, r8
        jae     4f
3:      mov     r9d, dword ptr [rdx + rax*4]
        mov     dword ptr [rcx + rax*4], r9d
        inc     rax
        cmp     rax, r8
        jb      3b
4:      vzeroupper
        ret

// ===========================================================================
// expf_vec and SiLU
// ===========================================================================

// SILU8 g, u : 8 lanes of  g[i] = (z * s) * u[i],  s = 1/(1 + expf(-z)), z = g[i]
// (exactly the C expression in llama.c / swiglu.c; with expf bit-exact and IEEE
// divide, every lane is bit-identical to the C result).
// Uses ymm0-ymm6.
.macro SILU8 gp, up
        vmovups   ymm0, ymmword ptr \gp                  // z
        vxorps    ymm0, ymm0, ymmword ptr [rip + C_SIGN] // -z
        EXP8      ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6   // expf(-z)
        vaddps    ymm0, ymm0, ymmword ptr [rip + C_ONE]  // 1 + expf(-z)
        vmovaps   ymm1, ymmword ptr [rip + C_ONE]
        vdivps    ymm0, ymm1, ymm0                       // s = 1 / (...)
        vmulps    ymm0, ymm0, ymmword ptr \gp            // z * s
        vmulps    ymm0, ymm0, ymmword ptr \up            // (z * s) * u
        vmovups   ymmword ptr \gp, ymm0
.endm

.macro EXPV8 dp, sp
        vmovups   ymm0, ymmword ptr \sp
        EXP8      ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6
        vmovups   ymmword ptr \dp, ymm0
.endm

// Frame shared by silu_mul and expf_vec:
//   [rsp+0]  saved xmm6 (16 B, needs 16-byte alignment)
//   [rsp+16] 32-byte buffer A  (tail of g / src)
//   [rsp+48] 32-byte buffer B  (tail of u)
// The (n mod 8) tail is copied into the zero-padded buffers, processed with the
// same vector code, and copied back, so no load ever reads past the arrays.
#define EW_FRAME 88          // 8 (ret) + 88 = 96: rsp 16-aligned inside

// void silu_mul(float* g, const float* u, i64 n)
        .globl  silu_mul
        .p2align 4
silu_mul:
        sub     rsp, EW_FRAME
        vmovaps xmmword ptr [rsp], xmm6
        xor     eax, eax
        mov     r9, r8
        and     r9, -8
        jz      2f
1:      SILU8   "[rcx + rax*4]", "[rdx + rax*4]"
        add     rax, 8
        cmp     rax, r9
        jb      1b
2:      mov     r10, r8
        sub     r10, rax                        // tail length 0..7
        jz      5f
        vxorps  ymm0, ymm0, ymm0
        vmovups ymmword ptr [rsp + 16], ymm0
        vmovups ymmword ptr [rsp + 48], ymm0
        xor     r11d, r11d
3:      lea     r9, [rax + r11]
        vmovss  xmm0, dword ptr [rcx + r9*4]
        vmovss  dword ptr [rsp + 16 + r11*4], xmm0
        vmovss  xmm0, dword ptr [rdx + r9*4]
        vmovss  dword ptr [rsp + 48 + r11*4], xmm0
        inc     r11
        cmp     r11, r10
        jb      3b
        SILU8   "[rsp + 16]", "[rsp + 48]"
        xor     r11d, r11d
4:      lea     r9, [rax + r11]
        vmovss  xmm0, dword ptr [rsp + 16 + r11*4]
        vmovss  dword ptr [rcx + r9*4], xmm0
        inc     r11
        cmp     r11, r10
        jb      4b
5:      vmovaps xmm6, xmmword ptr [rsp]
        add     rsp, EW_FRAME
        vzeroupper
        ret

// void expf_vec(float* dst, const float* src, i64 n)   (dst may == src)
        .globl  expf_vec
        .p2align 4
expf_vec:
        sub     rsp, EW_FRAME
        vmovaps xmmword ptr [rsp], xmm6
        xor     eax, eax
        mov     r9, r8
        and     r9, -8
        jz      2f
1:      EXPV8   "[rcx + rax*4]", "[rdx + rax*4]"
        add     rax, 8
        cmp     rax, r9
        jb      1b
2:      mov     r10, r8
        sub     r10, rax
        jz      5f
        vxorps  ymm0, ymm0, ymm0
        vmovups ymmword ptr [rsp + 16], ymm0
        xor     r11d, r11d
3:      lea     r9, [rax + r11]
        vmovss  xmm0, dword ptr [rdx + r9*4]
        vmovss  dword ptr [rsp + 16 + r11*4], xmm0
        inc     r11
        cmp     r11, r10
        jb      3b
        EXPV8   "[rsp + 16]", "[rsp + 16]"
        xor     r11d, r11d
4:      lea     r9, [rax + r11]
        vmovss  xmm0, dword ptr [rsp + 16 + r11*4]
        vmovss  dword ptr [rcx + r9*4], xmm0
        inc     r11
        cmp     r11, r10
        jb      4b
5:      vmovaps xmm6, xmmword ptr [rsp]
        add     rsp, EW_FRAME
        vzeroupper
        ret

// ===========================================================================
// RoPE (rope.c, half-split convention: pair k of a head is (x[k], x[k+hd/2]))
//
//   inv_freq[k] = powf(base, -2.0f*k/hd)        (float exponent, float result)
//   ang         = (float)pos * inv_freq[k]       (float multiply)
//   y[k]        = x1*cosf(ang) - x2*sinf(ang)
//   y[k+hd/2]   = x2*cosf(ang) + x1*sinf(ang)
//
// powf is evaluated on the x87 FPU in 64-bit-mantissa precision and rounded once
// to float (= the correctly rounded value, which is also what the C library's
// powf returns: checked bit for bit). Windows starts threads with the x87
// precision control at 53 bits, so we switch it to 64 bits (control word bits
// 8-9 = 11b) around the computation. cosf/sinf are NOT correctly rounded in the C
// library (1 ulp off in ~0.02% of arguments), so they are exact ports of its code
// instead (sinf_c / cosf_c below) -- the tables are bit-identical to rope.c's.
// ===========================================================================

// void rope_inv_freq(float* inv, i64 head_dim, float base)
        .globl  rope_inv_freq
        .p2align 4
rope_inv_freq:
        sub     rsp, 24                         // [rsp] e, [rsp+4] base, [rsp+8] cw, [rsp+12] cw'
        vmovss  dword ptr [rsp + 4], xmm2
        fnstcw  word ptr [rsp + 8]
        movzx   eax, word ptr [rsp + 8]
        or      eax, 0x0300                     // precision control = 64-bit mantissa
        mov     word ptr [rsp + 12], ax
        fldcw   word ptr [rsp + 12]
        vcvtsi2ss xmm1, xmm1, rdx               // (float)hd
        mov     r8, rdx
        shr     r8, 1                           // half = hd/2
        xor     eax, eax
        test    r8, r8
        jz      2f
1:      vcvtsi2ss xmm0, xmm0, rax               // (float)k
        vmulss  xmm0, xmm0, dword ptr [rip + C_MTWO]    // -2.0f * (float)k
        vdivss  xmm0, xmm0, xmm1                // / (float)hd   -> exponent e (float)
        vmovss  dword ptr [rsp], xmm0
        // powf(base, e) = 2^(e*log2(base))
        fld     dword ptr [rsp]                 // st0 = e
        fld     dword ptr [rsp + 4]             // st0 = base, st1 = e
        fyl2x                                   // st0 = t = e*log2(base)
        fld     st(0)                           // st0 = t, st1 = t
        frndint                                 // st0 = n = round(t)
        fsub    st(1), st(0)                    // st1 = f = t - n, |f| <= 0.5
        fxch    st(1)                           // st0 = f, st1 = n
        f2xm1                                   // st0 = 2^f - 1
        fld1
        faddp   st(1), st(0)                    // st0 = 2^f
        fscale                                  // st0 = 2^f * 2^n
        fstp    st(1)                           // drop n
        fstp    dword ptr [rcx + rax*4]         // round to float, store
        inc     rax
        cmp     rax, r8
        jb      1b
2:      fldcw   word ptr [rsp + 8]              // restore the caller's control word
        add     rsp, 24
        ret

// ---------------------------------------------------------------------------
// sinf / cosf, bit-exact ports of the C library's (zig compiler_rt = musl
// sinf.c / cosf.c with __sindf, __cosdf, __rem_pio2f): the argument is widened to
// double, reduced by multiples of pi/2, and a short minimax polynomial in double
// is rounded to float once. Every double operation below is the C operation in
// the same order (zig compiles these without FMA contraction).
// Only finite x >= 0 below 2^28*pi/2 is handled (RoPE angles: pos * inv_freq).
// ---------------------------------------------------------------------------
        .section .rdata,"dr"
        .p2align 4
D_S1:      .quad 0xbfc5555554cbac77    // -0.16666666641626524
D_S2:      .quad 0x3f811110896efbb2    // 0.008333329385889463
D_S3:      .quad 0xbf2a00f9e2cae774    // -0.00019839334836096632
D_S4:      .quad 0x3ec6cd878c3b46a7    // 2.718311493989822e-06
D_C0:      .quad 0xbfdffffffd0c5e81    // -0.499999997251031
D_C1:      .quad 0x3fa55553e1053a42    // 0.04166662332373906
D_C2:      .quad 0xbf56c087e80f1e27    // -0.001388676377460993
D_C3:      .quad 0x3ef99342e0ee5069    // 2.439044879627741e-05
D_PIO2_1:  .quad 0x3ff921fb50000000    // first 25 bits of pi/2
D_PIO2_1T: .quad 0x3e5110b4611a6263    // pi/2 - D_PIO2_1
D_INVPIO2: .quad 0x3fe45f306dc9c883    // 2/pi
D_TOINT:   .quad 0x4338000000000000    // 1.5*2^52: x + TOINT - TOINT = rint(x)
D_PIO4:    .quad 0x3fe921fb60000000    // (double)(float)(pi/4)
D_NPIO4:   .quad 0xbfe921fb60000000
D_X1PIO2:  .quad 0x3ff921fb54442d18    // 1*pi/2
D_X2PIO2:  .quad 0x400921fb54442d18    // 2*pi/2
D_X3PIO2:  .quad 0x4012d97c7f3321d2    // 3*pi/2
D_X4PIO2:  .quad 0x401921fb54442d18    // 4*pi/2
D_ONE:     .quad 0x3ff0000000000000
D_SIGN:    .quad 0x8000000000000000, 0
F_SIGN:    .long 0x80000000, 0, 0, 0
        .text

// sindf_k: xmm0 (double x, |x| <~ pi/4) -> xmm0 = (float) sin. Clobbers xmm1-5.
//   z = x*x; w = z*z; r = S3 + z*S4; s = z*x;
//   (float)((x + s*(S1 + z*S2)) + s*w*r)
        .p2align 4
sindf_k:
        vmulsd  xmm1, xmm0, xmm0                        // z
        vmulsd  xmm2, xmm1, xmm1                        // w
        vmulsd  xmm3, xmm1, qword ptr [rip + D_S4]
        vaddsd  xmm3, xmm3, qword ptr [rip + D_S3]      // r = S3 + z*S4
        vmulsd  xmm4, xmm1, xmm0                        // s = z*x
        vmulsd  xmm5, xmm1, qword ptr [rip + D_S2]
        vaddsd  xmm5, xmm5, qword ptr [rip + D_S1]      // S1 + z*S2
        vmulsd  xmm5, xmm5, xmm4                        // s*(S1 + z*S2)
        vaddsd  xmm5, xmm5, xmm0                        // x + ...
        vmulsd  xmm4, xmm4, xmm2                        // s*w
        vmulsd  xmm4, xmm4, xmm3                        // (s*w)*r
        vaddsd  xmm0, xmm5, xmm4
        vcvtsd2ss xmm0, xmm0, xmm0
        ret

// cosdf_k: xmm0 (double x) -> xmm0 = (float) cos. Clobbers xmm1-5.
//   z = x*x; w = z*z; r = C2 + z*C3; (float)(((1 + z*C0) + w*C1) + (w*z)*r)
        .p2align 4
cosdf_k:
        vmulsd  xmm1, xmm0, xmm0                        // z
        vmulsd  xmm2, xmm1, xmm1                        // w
        vmulsd  xmm3, xmm1, qword ptr [rip + D_C3]
        vaddsd  xmm3, xmm3, qword ptr [rip + D_C2]      // r
        vmulsd  xmm4, xmm1, qword ptr [rip + D_C0]
        vaddsd  xmm4, xmm4, qword ptr [rip + D_ONE]     // 1 + z*C0
        vmulsd  xmm5, xmm2, qword ptr [rip + D_C1]
        vaddsd  xmm4, xmm4, xmm5                        // + w*C1
        vmulsd  xmm5, xmm2, xmm1                        // w*z
        vmulsd  xmm5, xmm5, xmm3                        // (w*z)*r
        vaddsd  xmm0, xmm4, xmm5
        vcvtsd2ss xmm0, xmm0, xmm0
        ret

// rem_pio2f_k: xmm0 = (double)x  ->  xmm0 = y = x - n*pi/2 (double), eax = n.
// Clobbers xmm1-3.  (musl __rem_pio2f, medium-size branch)
        .p2align 4
rem_pio2f_k:
        vmulsd  xmm1, xmm0, qword ptr [rip + D_INVPIO2]
        vaddsd  xmm1, xmm1, qword ptr [rip + D_TOINT]
        vsubsd  xmm1, xmm1, qword ptr [rip + D_TOINT]   // fn = rint(x*2/pi)
        vcvttsd2si eax, xmm1                            // n
1:      vmulsd  xmm2, xmm1, qword ptr [rip + D_PIO2_1]
        vsubsd  xmm3, xmm0, xmm2                        // x - fn*pio2_1
        vmulsd  xmm2, xmm1, qword ptr [rip + D_PIO2_1T]
        vsubsd  xmm3, xmm3, xmm2                        //   - fn*pio2_1t
        vucomisd xmm3, qword ptr [rip + D_NPIO4]
        jb      2f                                      // y < -pi/4
        vucomisd xmm3, qword ptr [rip + D_PIO4]
        ja      3f                                      // y > pi/4
        vmovapd xmm0, xmm3
        ret
2:      dec     eax                                     // (only with directed rounding)
        vsubsd  xmm1, xmm1, qword ptr [rip + D_ONE]
        jmp     4f
3:      inc     eax
        vaddsd  xmm1, xmm1, qword ptr [rip + D_ONE]
4:      vmulsd  xmm2, xmm1, qword ptr [rip + D_PIO2_1]
        vsubsd  xmm3, xmm0, xmm2
        vmulsd  xmm2, xmm1, qword ptr [rip + D_PIO2_1T]
        vsubsd  xmm0, xmm3, xmm2
        ret

// sinf_c: xmm0 = float x (>= 0) -> xmm0 = sinf(x).  Clobbers rax, xmm1-5.
        .p2align 4
sinf_c:
        vmovd   eax, xmm0
        vcvtss2sd xmm0, xmm0, xmm0
        cmp     eax, 0x3f490fda                 // |x| <~ pi/4
        ja      1f
        cmp     eax, 0x39800000                 // |x| < 2^-12: sin x = x
        jae     sindf_k
        vcvtsd2ss xmm0, xmm0, xmm0
        ret
1:      cmp     eax, 0x407b53d1                 // <~ 5pi/4
        ja      3f
        cmp     eax, 0x4016cbe3                 // <~ 3pi/4
        ja      2f
        vsubsd  xmm0, xmm0, qword ptr [rip + D_X1PIO2]
        jmp     cosdf_k                         // cos(x - pi/2)
2:      vsubsd  xmm0, xmm0, qword ptr [rip + D_X2PIO2]
        vxorpd  xmm0, xmm0, xmmword ptr [rip + D_SIGN]
        jmp     sindf_k                         // sin(-(x - pi))
3:      cmp     eax, 0x40e231d5                 // <~ 9pi/4
        ja      5f
        cmp     eax, 0x40afeddf                 // <~ 7pi/4
        ja      4f
        vsubsd  xmm0, xmm0, qword ptr [rip + D_X3PIO2]
        call    cosdf_k
        vxorps  xmm0, xmm0, xmmword ptr [rip + F_SIGN]
        ret                                     // -cos(x - 3pi/2)
4:      vsubsd  xmm0, xmm0, qword ptr [rip + D_X4PIO2]
        jmp     sindf_k                         // sin(x - 2pi)
5:      call    rem_pio2f_k
        and     eax, 3
        jz      sindf_k                         // 0: sin y
        cmp     eax, 2
        jb      cosdf_k                         // 1: cos y
        je      6f
        call    cosdf_k                         // 3: -cos y
        vxorps  xmm0, xmm0, xmmword ptr [rip + F_SIGN]
        ret
6:      vxorpd  xmm0, xmm0, xmmword ptr [rip + D_SIGN]
        jmp     sindf_k                         // 2: sin(-y)

// cosf_c: xmm0 = float x (>= 0) -> xmm0 = cosf(x).  Clobbers rax, xmm1-5.
        .p2align 4
cosf_c:
        vmovd   eax, xmm0
        vcvtss2sd xmm0, xmm0, xmm0
        cmp     eax, 0x3f490fda
        ja      1f
        cmp     eax, 0x39800000
        jae     cosdf_k
        vmovss  xmm0, dword ptr [rip + C_ONE]   // |x| < 2^-12: 1
        ret
1:      cmp     eax, 0x407b53d1
        ja      3f
        cmp     eax, 0x4016cbe3
        ja      2f
        vmovsd  xmm1, qword ptr [rip + D_X1PIO2]
        vsubsd  xmm0, xmm1, xmm0
        jmp     sindf_k                         // sin(pi/2 - x)
2:      vsubsd  xmm0, xmm0, qword ptr [rip + D_X2PIO2]
        call    cosdf_k
        vxorps  xmm0, xmm0, xmmword ptr [rip + F_SIGN]
        ret                                     // -cos(x - pi)
3:      cmp     eax, 0x40e231d5
        ja      5f
        cmp     eax, 0x40afeddf
        ja      4f
        vsubsd  xmm0, xmm0, qword ptr [rip + D_X3PIO2]
        jmp     sindf_k                         // sin(x - 3pi/2)
4:      vsubsd  xmm0, xmm0, qword ptr [rip + D_X4PIO2]
        jmp     cosdf_k                         // cos(x - 2pi)
5:      call    rem_pio2f_k
        and     eax, 3
        jz      cosdf_k                         // 0: cos y
        cmp     eax, 2
        jb      6f
        je      7f
        jmp     sindf_k                         // 3: sin y
6:      vxorpd  xmm0, xmm0, xmmword ptr [rip + D_SIGN]
        jmp     sindf_k                         // 1: sin(-y)
7:      call    cosdf_k                         // 2: -cos y
        vxorps  xmm0, xmm0, xmmword ptr [rip + F_SIGN]
        ret

// void rope_table(float* cs, const float* inv, i64 half, i64 pos)
//   cs[k] = cosf(ang), cs[half+k] = sinf(ang), ang = (float)pos * inv[k]
        .globl  rope_table
        .p2align 4
rope_table:
        push    rbx
        push    rsi
        push    rdi
        push    r12
        sub     rsp, 8
        mov     rbx, rcx                        // cos half
        lea     rsi, [rcx + r8*4]               // sin half
        mov     rdi, rdx                        // inv
        mov     r12, r8
        vcvtsi2ss xmm0, xmm0, r9                // (float)pos (exact below 2^24)
        vmovss  dword ptr [rsp], xmm0
        xor     ecx, ecx
        test    r12, r12
        jz      2f
1:      vmovss  xmm0, dword ptr [rsp]
        vmulss  xmm0, xmm0, dword ptr [rdi + rcx*4]     // ang (float multiply, like C)
        vmovss  dword ptr [rsp + 4], xmm0
        call    cosf_c
        vmovss  dword ptr [rbx + rcx*4], xmm0
        vmovss  xmm0, dword ptr [rsp + 4]
        call    sinf_c
        vmovss  dword ptr [rsi + rcx*4], xmm0
        inc     rcx
        cmp     rcx, r12
        jb      1b
2:      add     rsp, 8
        pop     r12
        pop     rdi
        pop     rsi
        pop     rbx
        ret

// rope_core (internal): rotate n_heads heads of one row in place.
//   in: rcx = v, rdx = n_heads, r8 = head_dim, r9 = cs (cos[half], sin[half])
//   clobbers rax, r10, r11, ymm0-ymm5; leaves rcx pointing past the row.
        .p2align 4
rope_core:
        mov     r10, r8
        shr     r10, 1                          // half
        lea     r11, [r9 + r10*4]               // sin table
        test    rdx, rdx
        jz      5f
1:      xor     eax, eax                        // k = 0
        push    rdx
        mov     rdx, r10
        and     rdx, -8                         // vector part of half
        jz      3f
2:      vmovups ymm0, ymmword ptr [rcx + rax*4]         // x1 = v[k..k+8)
        lea     rax, [rax + r10]
        vmovups ymm1, ymmword ptr [rcx + rax*4]         // x2 = v[k+half..)
        sub     rax, r10
        vmovups ymm2, ymmword ptr [r9 + rax*4]          // c
        vmovups ymm3, ymmword ptr [r11 + rax*4]         // s
        vmulps  ymm4, ymm1, ymm3                        // x2*s
        vfmsub231ps ymm4, ymm0, ymm2                    // y1 = x1*c - x2*s
        vmulps  ymm5, ymm0, ymm3                        // x1*s
        vfmadd231ps ymm5, ymm1, ymm2                    // y2 = x2*c + x1*s
        vmovups ymmword ptr [rcx + rax*4], ymm4
        lea     rax, [rax + r10]
        vmovups ymmword ptr [rcx + rax*4], ymm5
        sub     rax, r10
        add     rax, 8
        cmp     rax, rdx
        jb      2b
3:      cmp     rax, r10                        // scalar tail (half not a multiple of 8)
        jae     4f
        vmovss  xmm0, dword ptr [rcx + rax*4]
        lea     rdx, [rax + r10]
        vmovss  xmm1, dword ptr [rcx + rdx*4]
        vmovss  xmm2, dword ptr [r9 + rax*4]
        vmovss  xmm3, dword ptr [r11 + rax*4]
        vmulss  xmm4, xmm1, xmm3
        vfmsub231ss xmm4, xmm0, xmm2
        vmulss  xmm5, xmm0, xmm3
        vfmadd231ss xmm5, xmm1, xmm2
        vmovss  dword ptr [rcx + rax*4], xmm4
        vmovss  dword ptr [rcx + rdx*4], xmm5
        inc     rax
        jmp     3b
4:      pop     rdx
        lea     rcx, [rcx + r8*4]               // next head
        dec     rdx
        jnz     1b
5:      ret

// void rope_apply(float* v, i64 n_heads, i64 head_dim, const float* cs)
        .globl  rope_apply
        .p2align 4
rope_apply:
        sub     rsp, 8                          // keep the push in rope_core harmless
        call    rope_core
        add     rsp, 8
        vzeroupper
        ret

// void rope_apply_rows(float* v, i64 n_heads, i64 head_dim, const float* cs, i64 rows)
//   row r: v + r*n_heads*head_dim, table cs + r*head_dim
        .globl  rope_apply_rows
        .p2align 4
rope_apply_rows:
        push    rbx
        push    rsi
        sub     rsp, 8
        mov     rbx, qword ptr [rsp + 24 + 40]  // rows (5th arg; 24 = our pushes+sub)
        mov     rsi, rdx                        // n_heads (rope_core consumes rdx)
        test    rbx, rbx
        jle     2f
1:      mov     rdx, rsi
        call    rope_core                       // rcx advances by one row
        lea     r9, [r9 + r8*4]                 // next position's table
        dec     rbx
        jnz     1b
2:      add     rsp, 8
        pop     rsi
        pop     rbx
        vzeroupper
        ret

// void rope_pos(float* v, i64 n_heads, i64 head_dim, float base, i64 pos)
// One-shot convenience (computes inv_freq and the table on the stack). hd <= 256.
#define RP_FRAME (32 + 512 + 1024)              // shadow, inv[128], cs[256]
        .globl  rope_pos
        .p2align 4
rope_pos:
        push    rbx
        push    rsi
        push    rdi
        sub     rsp, RP_FRAME                   // 8 + 24 + 1568 = 1600: aligned
        cmp     r8, 256
        ja      9f
        mov     rbx, rcx                        // v
        mov     rsi, rdx                        // n_heads
        mov     rdi, r8                         // head_dim
        lea     rcx, [rsp + 32]                 // inv
        mov     rdx, rdi
        vmovaps xmm2, xmm3                      // base: 4th arg -> 3rd position
        call    rope_inv_freq
        lea     rcx, [rsp + 32 + 512]           // cs
        lea     rdx, [rsp + 32]
        mov     r8, rdi
        shr     r8, 1
        mov     r9, qword ptr [rsp + RP_FRAME + 24 + 40]    // pos (5th arg)
        call    rope_table
        mov     rcx, rbx
        mov     rdx, rsi
        mov     r8, rdi
        lea     r9, [rsp + 32 + 512]
        call    rope_apply
9:      add     rsp, RP_FRAME
        pop     rdi
        pop     rsi
        pop     rbx
        ret

// ===========================================================================
// KV cache layout (owned by this module: only ops.S/decode.S read or write the
// cache CONTENTS; kv_create only allocates cap*kv_dim floats per layer)
//
//   V: row-major, slot s at v + s*kv_dim             (what the weighted sum wants)
//   K: if cap % 8 == 0 (every real model: cap = max_seq or a window like 4096):
//        BLOCKED - slots 8b..8b+7 form one block of 8*kv_dim floats stored
//        channel-major:  K[s][c] lives at k + (s>>3)*8*kv_dim + c*8 + (s&7)
//      else: row-major like V.
// Why: C computes every score as ONE serial FMA chain over d (dot = fma(q[d],
// k[d], dot)). To be bit-exact we must keep that chain per key -- so instead of
// splitting one dot product across vector lanes, the 8 lanes hold 8 DIFFERENT
// keys, each running its own serial chain: one vfmadd231ps advances 8 dot
// products by one d, reading 32 contiguous bytes of a block. The blocked layout
// makes those 8 values (same channel, 8 consecutive slots) contiguous.
// Copying a cache prefix of `len` positions = copying ceil(len/8) whole blocks.
// ===========================================================================

// void kv_put(float* cache, i64 cap, i64 slot, const float* row, i64 kv_dim)
//   store one K row into slot `slot` (layout above).  5th arg kv_dim at [rsp+40].
        .globl  kv_put
        .p2align 4
kv_put:
        mov     r10, qword ptr [rsp + 40]       // kv_dim
        test    rdx, 7
        jnz     .Lkp_rowmajor
        mov     rax, r8
        shr     rax, 3                          // block
        imul    rax, r10
        shl     rax, 5                          // * 8 lanes * 4 bytes
        add     rcx, rax
        and     r8d, 7
        lea     rcx, [rcx + r8*4]               // lane within the block
        xor     eax, eax
1:      mov     edx, dword ptr [r9 + rax*4]     // channel c -> stride 8 floats
        mov     dword ptr [rcx], edx
        add     rcx, 32
        inc     rax
        cmp     rax, r10
        jb      1b
        ret
.Lkp_rowmajor:
        imul    r8, r10
        lea     rcx, [rcx + r8*4]
        xor     eax, eax
2:      mov     edx, dword ptr [r9 + rax*4]
        mov     dword ptr [rcx + rax*4], edx
        inc     rax
        cmp     rax, r10
        jb      2b
        ret

// ===========================================================================
// Attention (llama.c decode_attention), for one or many query rows.
//
// Task t = row*n_heads + head. For each task:
//   pos = pos0 + row ; lo = window > 0 ? max(0, pos-window+1) : 0 (gqa_window_lo)
//   s_j = (q_h . k_j) * (1/sqrtf(hd))     j = lo..pos, key j in slot j % cap
//   m = max s_j ; e_j = expf(s_j - m) ; sum = e_lo + ... + e_pos ; inv = 1/sum
//   o_h = sum_j (e_j*inv) * v_j          (j ascending, one FMA per element)
// The query head h reads kv head h / (n_heads/n_kv) (GQA). Every float operation
// matches the clang-compiled C (serial FMA dot, serial float sum, bit-exact
// expf), so the output is BIT-IDENTICAL to decode_attention.
//
// Scores live in the task's scratch at index (j - glo), where glo = lo rounded
// down to a multiple of 8 for the blocked K layout (0 extra for row-major); the
// slack lanes at both ends are set to -inf, so the vector max ignores them and
// their exp is exactly 0.
// ===========================================================================

// void attention(AttnArgs* a)
        .globl  attention
        .p2align 4
attention:
        push    rbx
        sub     rsp, 32                         // 8 + 8 + 32 = 48: aligned for calls
        mov     rbx, rcx
        mov     rax, qword ptr [rbx + AT_NHEADS]
        cqo
        idiv    qword ptr [rbx + AT_NKV]
        mov     qword ptr [rbx + AT_GROUP], rax         // heads per kv head
        mov     rax, qword ptr [rbx + AT_HD]
        vcvtsi2ss xmm0, xmm0, rax
        vsqrtss xmm0, xmm0, xmm0
        vmovss  xmm1, dword ptr [rip + C_ONE]
        vdivss  xmm1, xmm1, xmm0                        // scale = 1.0f / sqrtf((float)hd)
        vmovss  dword ptr [rbx + AT_SCALE], xmm1
        mov     r8, qword ptr [rbx + AT_NROWS]
        imul    r8, qword ptr [rbx + AT_NHEADS]         // tasks
        mov     rax, qword ptr [rbx + AT_POS0]
        add     rax, qword ptr [rbx + AT_NROWS]         // ~ keys per task (upper bound)
        imul    rax, r8
        imul    rax, qword ptr [rbx + AT_HD]            // work estimate
        cmp     r8, 2
        jb      1f
        cmp     rax, ATTN_THREAD_MIN
        jb      1f
        lea     rcx, [rip + attn_tasks]                 // pool_run(fn, ctx, n, grain=1)
        mov     rdx, rbx
        mov     r9d, 1
        call    pool_run
        jmp     2f
1:      mov     rcx, rbx                                // attn_tasks(ctx, 0, tasks)
        xor     edx, edx
        call    attn_tasks
2:      add     rsp, 32
        pop     rbx
        vzeroupper
        ret

// attn_tasks(AttnArgs* a, i64 t0, i64 t1)  -- pool_run worker, runs tasks [t0, t1)
//
// Frame: 8 pushes, then AT_FRAME bytes: [rsp+0..95] locals, [rsp+96..255] xmm6-15.
#define AL_VBASE  0      // vc + kvh*hd*4
#define AL_SLOT0  8      // lo % cap (slot of the first key)
#define AL_QEND   16     // qh + hd floats
#define AL_OH     24     // output pointer of this task
#define AL_OFF    32     // index of key lo in the scores (lo - glo)
#define AL_P      40     // padded score count (multiple of 8)
#define AL_GCNT   48     // blocks to score (blocked layout)
#define AL_NBR    56     // blocks in the ring (cap/8)
#define AL_LO     64
#define AT_FRAME  264    // 8 + 64 + 264 = 336: 16-aligned
//
// Registers during a task:
//   rbx = args   r12 = t   r13 = t1   r14 = cnt (keys lo..pos)
//   rdi = qh     rsi = score scratch   r11 = cap
//   phase 1: r15 = K base of this kv head, rbp = block stride (8*kv_dim*4)
//   phase 3: rbp = V row stride (kv_dim*4)
        .p2align 4
attn_tasks:
        push    rbx
        push    rbp
        push    rsi
        push    rdi
        push    r12
        push    r13
        push    r14
        push    r15
        sub     rsp, AT_FRAME
        vmovaps xmmword ptr [rsp + 96], xmm6
        vmovaps xmmword ptr [rsp + 112], xmm7
        vmovaps xmmword ptr [rsp + 128], xmm8
        vmovaps xmmword ptr [rsp + 144], xmm9
        vmovaps xmmword ptr [rsp + 160], xmm10
        vmovaps xmmword ptr [rsp + 176], xmm11
        vmovaps xmmword ptr [rsp + 192], xmm12
        vmovaps xmmword ptr [rsp + 208], xmm13
        vmovaps xmmword ptr [rsp + 224], xmm14
        vmovaps xmmword ptr [rsp + 240], xmm15
        mov     rbx, rcx
        mov     r12, rdx
        mov     r13, r8
        cmp     r12, r13
        jae     .Lat_ret

.Lat_task:
        // ---- task setup ----
        mov     rax, r12
        xor     edx, edx
        div     qword ptr [rbx + AT_NHEADS]     // rax = row, rdx = head
        mov     rcx, rdx                        // rcx = head
        add     rax, qword ptr [rbx + AT_POS0]  // rax = pos
        xor     r8d, r8d                        // lo = 0
        mov     rdx, qword ptr [rbx + AT_WINDOW]
        test    rdx, rdx
        jle     1f
        mov     r9, rax
        sub     r9, rdx
        inc     r9                              // pos - window + 1
        test    r9, r9
        jle     1f
        mov     r8, r9                          // lo = pos - window + 1
1:      mov     qword ptr [rsp + AL_LO], r8
        mov     r14, rax
        sub     r14, r8
        inc     r14                             // cnt = pos - lo + 1
        mov     r11, qword ptr [rbx + AT_CAP]
        mov     rax, r8
        xor     edx, edx
        div     r11
        mov     qword ptr [rsp + AL_SLOT0], rdx // slot of key lo
        mov     rax, rcx
        xor     edx, edx
        div     qword ptr [rbx + AT_GROUP]      // rax = kv head
        mov     rcx, qword ptr [rbx + AT_HD]
        imul    rax, rcx                        // kvh*hd (first channel of the kv head)
        mov     rdx, qword ptr [rbx + AT_VC]
        lea     rdx, [rdx + rax*4]
        mov     qword ptr [rsp + AL_VBASE], rdx
        mov     r15, qword ptr [rbx + AT_KC]
        test    r11, 7
        jnz     2f
        shl     rax, 3                          // blocked: channel stride is 8 floats
2:      lea     r15, [r15 + rax*4]              // K base of this kv head
        mov     rdi, r12
        imul    rdi, rcx
        shl     rdi, 2                          // t*hd*4 (q and o have the same layout)
        mov     rax, qword ptr [rbx + AT_O]
        add     rax, rdi
        mov     qword ptr [rsp + AL_OH], rax
        add     rdi, qword ptr [rbx + AT_Q]     // qh
        lea     rax, [rdi + rcx*4]
        mov     qword ptr [rsp + AL_QEND], rax
        mov     rsi, r12
        imul    rsi, qword ptr [rbx + AT_SCSTRIDE]
        shl     rsi, 2
        add     rsi, qword ptr [rbx + AT_SC]    // scores of this task
        vbroadcastss ymm7, dword ptr [rbx + AT_SCALE]
        mov     rbp, qword ptr [rbx + AT_NKV]
        imul    rbp, rcx
        shl     rbp, 2                          // kv_dim*4 = row stride
        test    r11, 7
        jnz     .Lp1_rowmajor

        // ---- phase 1, blocked K: 8 keys per vector, serial FMA chain per lane ----
        shl     rbp, 3                          // block stride = 8 rows
        mov     rax, qword ptr [rsp + AL_LO]
        mov     rdx, rax
        and     rdx, -8                         // glo
        sub     rax, rdx
        mov     qword ptr [rsp + AL_OFF], rax   // off = lo - glo
        mov     rax, r14
        add     rax, qword ptr [rsp + AL_OFF]
        add     rax, 7
        shr     rax, 3                          // blocks covering glo..pos
        mov     qword ptr [rsp + AL_GCNT], rax
        shl     rax, 3
        mov     qword ptr [rsp + AL_P], rax
        mov     rax, r11
        shr     rax, 3
        mov     qword ptr [rsp + AL_NBR], rax
        mov     r9, qword ptr [rsp + AL_SLOT0]
        shr     r9, 3                           // ring block of glo (slot0 - off is 8-aligned)
        xor     r10d, r10d                      // i = block counter
.Lb_loop:
        mov     rax, qword ptr [rsp + AL_GCNT]
        sub     rax, r10
        jbe     .Lb_done
        cmp     rax, 8
        jb      .Lb_one
        lea     rax, [r9 + 8]
        cmp     rax, qword ptr [rsp + AL_NBR]
        ja      .Lb_one                         // the 8 blocks would wrap the ring
        // 8 blocks (64 keys): 8 accumulators = 8 x 8 independent chains
        mov     rax, r9
        imul    rax, rbp
        add     rax, r15                        // block 0
        lea     rcx, [rax + rbp*2]
        add     rcx, rbp                        // block 3
        lea     rdx, [rcx + rbp*2]              // block 5
        vxorps  ymm8, ymm8, ymm8
        vxorps  ymm9, ymm9, ymm9
        vxorps  ymm10, ymm10, ymm10
        vxorps  ymm11, ymm11, ymm11
        vxorps  ymm12, ymm12, ymm12
        vxorps  ymm13, ymm13, ymm13
        vxorps  ymm14, ymm14, ymm14
        vxorps  ymm15, ymm15, ymm15
        mov     r8, rdi
3:      vbroadcastss ymm0, dword ptr [r8]               // q[d] in all lanes
        vfmadd231ps ymm8,  ymm0, ymmword ptr [rax]              // keys of block 0
        vfmadd231ps ymm9,  ymm0, ymmword ptr [rax + rbp]        // block 1
        vfmadd231ps ymm10, ymm0, ymmword ptr [rax + rbp*2]      // block 2
        vfmadd231ps ymm11, ymm0, ymmword ptr [rcx]              // block 3
        vfmadd231ps ymm12, ymm0, ymmword ptr [rcx + rbp]        // block 4
        vfmadd231ps ymm13, ymm0, ymmword ptr [rdx]              // block 5
        vfmadd231ps ymm14, ymm0, ymmword ptr [rdx + rbp]        // block 6
        vfmadd231ps ymm15, ymm0, ymmword ptr [rdx + rbp*2]      // block 7
        add     r8, 4                           // next d
        add     rax, 32                         // next channel row of each block
        add     rcx, 32
        add     rdx, 32
        cmp     r8, qword ptr [rsp + AL_QEND]
        jb      3b
        mov     rax, r10
        shl     rax, 5
        add     rax, rsi                        // &scores[i*8]
        vmulps  ymm8, ymm8, ymm7                // s = dot * scale
        vmulps  ymm9, ymm9, ymm7
        vmulps  ymm10, ymm10, ymm7
        vmulps  ymm11, ymm11, ymm7
        vmulps  ymm12, ymm12, ymm7
        vmulps  ymm13, ymm13, ymm7
        vmulps  ymm14, ymm14, ymm7
        vmulps  ymm15, ymm15, ymm7
        vmovups ymmword ptr [rax], ymm8
        vmovups ymmword ptr [rax + 32], ymm9
        vmovups ymmword ptr [rax + 64], ymm10
        vmovups ymmword ptr [rax + 96], ymm11
        vmovups ymmword ptr [rax + 128], ymm12
        vmovups ymmword ptr [rax + 160], ymm13
        vmovups ymmword ptr [rax + 192], ymm14
        vmovups ymmword ptr [rax + 224], ymm15
        add     r10, 8
        add     r9, 8
        cmp     r9, qword ptr [rsp + AL_NBR]
        jb      .Lb_loop
        xor     r9d, r9d                        // wrapped to block 0
        jmp     .Lb_loop
.Lb_one:                                        // one block (8 keys), latency-bound
        mov     rax, r9
        imul    rax, rbp
        add     rax, r15
        vxorps  ymm8, ymm8, ymm8
        mov     r8, rdi
4:      vbroadcastss ymm0, dword ptr [r8]
        vfmadd231ps ymm8, ymm0, ymmword ptr [rax]
        add     r8, 4
        add     rax, 32
        cmp     r8, qword ptr [rsp + AL_QEND]
        jb      4b
        vmulps  ymm8, ymm8, ymm7
        mov     rax, r10
        shl     rax, 5
        vmovups ymmword ptr [rsi + rax], ymm8
        inc     r10
        inc     r9
        cmp     r9, qword ptr [rsp + AL_NBR]
        jb      .Lb_loop
        xor     r9d, r9d
        jmp     .Lb_loop
.Lb_done:
        // -inf into the slack lanes: [0, off) and [off+cnt, P)
        vbroadcastss ymm0, dword ptr [rip + C_NEGINF]
        xor     eax, eax
5:      cmp     rax, qword ptr [rsp + AL_OFF]
        jae     6f
        vmovss  dword ptr [rsi + rax*4], xmm0
        inc     rax
        jmp     5b
6:      mov     rax, qword ptr [rsp + AL_OFF]
        add     rax, r14
7:      cmp     rax, qword ptr [rsp + AL_P]
        jae     .Lp2
        vmovss  dword ptr [rsi + rax*4], xmm0
        inc     rax
        jmp     7b

        // ---- phase 1, row-major K (cap not a multiple of 8): scalar serial dots ----
.Lp1_rowmajor:
        mov     qword ptr [rsp + AL_OFF], 0
        lea     rax, [r14 + 7]
        and     rax, -8
        mov     qword ptr [rsp + AL_P], rax
        mov     r9, qword ptr [rsp + AL_SLOT0]
        xor     r10d, r10d
8:      mov     rax, r9
        imul    rax, rbp
        add     rax, r15                        // k row
        vxorps  xmm8, xmm8, xmm8
        mov     r8, rdi
9:      vmovss  xmm0, dword ptr [r8]
        vfmadd231ss xmm8, xmm0, dword ptr [rax]         // dot = fma(q[d], k[d], dot)
        add     r8, 4
        add     rax, 4
        cmp     r8, qword ptr [rsp + AL_QEND]
        jb      9b
        vmulss  xmm8, xmm8, xmm7
        vmovss  dword ptr [rsi + r10*4], xmm8
        inc     r10
        inc     r9
        cmp     r9, r11
        jb      10f
        xor     r9d, r9d
10:     cmp     r10, r14
        jb      8b
        vbroadcastss ymm0, dword ptr [rip + C_NEGINF]
11:     cmp     r10, qword ptr [rsp + AL_P]
        jae     .Lp2
        vmovss  dword ptr [rsi + r10*4], xmm0
        inc     r10
        jmp     11b

        // ---- phase 2: m = max, e = expf(s - m) (vector), sum serially ----
.Lp2:
        mov     rcx, qword ptr [rsp + AL_P]
        vbroadcastss ymm6, dword ptr [rip + C_NEGINF]
        xor     eax, eax
12:     vmaxps  ymm6, ymm6, ymmword ptr [rsi + rax*4]
        add     rax, 8
        cmp     rax, rcx
        jb      12b
        vextractf128 xmm0, ymm6, 1                      // horizontal max
        vmaxps  xmm0, xmm0, xmm6
        vpermilps xmm1, xmm0, 0x4E
        vmaxps  xmm0, xmm0, xmm1
        vpermilps xmm1, xmm0, 0xB1
        vmaxps  xmm0, xmm0, xmm1
        vbroadcastss ymm15, xmm0                        // m
        xor     r10d, r10d
13:     vmovups ymm0, ymmword ptr [rsi + r10*4]
        vsubps  ymm0, ymm0, ymm15
        EXP8    ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6
        vmovups ymmword ptr [rsi + r10*4], ymm0
        add     r10, 8
        cmp     r10, rcx
        jb      13b
        mov     rax, qword ptr [rsp + AL_OFF]
        lea     rsi, [rsi + rax*4]              // rsi = &e[lo] from here on
        vxorps  xmm14, xmm14, xmm14             // sum = 0, then += e_j in order
        xor     eax, eax
14:     vaddss  xmm14, xmm14, dword ptr [rsi + rax*4]
        inc     rax
        cmp     rax, r14
        jb      14b
        vmovss  xmm1, dword ptr [rip + C_ONE]
        vdivss  xmm1, xmm1, xmm14                       // inv = 1/sum

        // ---- phase 3: o = sum_j (e_j*inv) * v_j ----
        mov     rbp, qword ptr [rbx + AT_NKV]
        imul    rbp, qword ptr [rbx + AT_HD]
        shl     rbp, 2                          // V row stride
        mov     r9, qword ptr [rsp + AL_SLOT0]
        xor     r10d, r10d
        mov     r8, qword ptr [rsp + AL_VBASE]
        cmp     qword ptr [rbx + AT_HD], 64
        jne     .Lp3_generic
        vxorps  ymm8, ymm8, ymm8
        vxorps  ymm9, ymm9, ymm9
        vxorps  ymm10, ymm10, ymm10
        vxorps  ymm11, ymm11, ymm11
        vxorps  ymm12, ymm12, ymm12
        vxorps  ymm13, ymm13, ymm13
        vxorps  ymm14, ymm14, ymm14
        vxorps  ymm15, ymm15, ymm15
15:     vmulss  xmm0, xmm1, dword ptr [rsi + r10*4]     // w = e_j * inv
        vbroadcastss ymm0, xmm0
        mov     rax, r9
        imul    rax, rbp
        add     rax, r8                                 // v row of key j
        vfmadd231ps ymm8,  ymm0, ymmword ptr [rax]      // o = fma(w, v, o)
        vfmadd231ps ymm9,  ymm0, ymmword ptr [rax + 32]
        vfmadd231ps ymm10, ymm0, ymmword ptr [rax + 64]
        vfmadd231ps ymm11, ymm0, ymmword ptr [rax + 96]
        vfmadd231ps ymm12, ymm0, ymmword ptr [rax + 128]
        vfmadd231ps ymm13, ymm0, ymmword ptr [rax + 160]
        vfmadd231ps ymm14, ymm0, ymmword ptr [rax + 192]
        vfmadd231ps ymm15, ymm0, ymmword ptr [rax + 224]
        inc     r9
        cmp     r9, r11
        jb      16f
        xor     r9d, r9d
16:     inc     r10
        cmp     r10, r14
        jb      15b
        mov     rax, qword ptr [rsp + AL_OH]
        vmovups ymmword ptr [rax], ymm8
        vmovups ymmword ptr [rax + 32], ymm9
        vmovups ymmword ptr [rax + 64], ymm10
        vmovups ymmword ptr [rax + 96], ymm11
        vmovups ymmword ptr [rax + 128], ymm12
        vmovups ymmword ptr [rax + 160], ymm13
        vmovups ymmword ptr [rax + 192], ymm14
        vmovups ymmword ptr [rax + 224], ymm15
        jmp     .Lat_next

.Lp3_generic:                                   // any head_dim (multiple of 8): o in memory
        mov     rdx, qword ptr [rsp + AL_OH]
        mov     rdi, qword ptr [rbx + AT_HD]
        shl     rdi, 2                          // bytes per head (qh is not needed any more)
        vxorps  ymm2, ymm2, ymm2
        xor     eax, eax
17:     vmovups ymmword ptr [rdx + rax], ymm2   // o_h = 0
        add     rax, 32
        cmp     rax, rdi
        jb      17b
18:     vmulss  xmm0, xmm1, dword ptr [rsi + r10*4]     // w = e_j * inv
        vbroadcastss ymm0, xmm0
        mov     rax, r9
        imul    rax, rbp
        add     rax, r8                                 // v row of key j
        xor     ecx, ecx
19:     vmovups ymm2, ymmword ptr [rdx + rcx]
        vfmadd231ps ymm2, ymm0, ymmword ptr [rax + rcx] // o += w * v
        vmovups ymmword ptr [rdx + rcx], ymm2
        add     rcx, 32
        cmp     rcx, rdi
        jb      19b
        inc     r9
        cmp     r9, r11
        jb      20f
        xor     r9d, r9d
20:     inc     r10
        cmp     r10, r14
        jb      18b

.Lat_next:
        inc     r12
        cmp     r12, r13
        jb      .Lat_task
.Lat_ret:
        vmovaps xmm6, xmmword ptr [rsp + 96]
        vmovaps xmm7, xmmword ptr [rsp + 112]
        vmovaps xmm8, xmmword ptr [rsp + 128]
        vmovaps xmm9, xmmword ptr [rsp + 144]
        vmovaps xmm10, xmmword ptr [rsp + 160]
        vmovaps xmm11, xmmword ptr [rsp + 176]
        vmovaps xmm12, xmmword ptr [rsp + 192]
        vmovaps xmm13, xmmword ptr [rsp + 208]
        vmovaps xmm14, xmmword ptr [rsp + 224]
        vmovaps xmm15, xmmword ptr [rsp + 240]
        add     rsp, AT_FRAME
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
