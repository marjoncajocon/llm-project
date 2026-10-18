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
//          bit-exact with llama.c decode_attention; threaded with pool_run
//          (key-split decode / GQA-shared row tasks, see "Attention" below).
//          Keeps no static state (everything lives on its stack), but like
//          pool_run it must not be called from two threads at once.
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
// What has to be computed, per query row (position pos) and query head h:
//   lo  = window > 0 ? max(0, pos-window+1) : 0          (gqa_window_lo)
//   s_j = (q_h . k_j) * (1/sqrtf(hd))     j = lo..pos, key j lives in slot j % cap
//   m   = max s_j ; e_j = expf(s_j - m) ; sum = e_lo + ... + e_pos ; inv = 1/sum
//   o_h = sum_j (e_j*inv) * v_j          (j ascending, one FMA per element)
// Query head h reads kv head h / group (GQA, group = n_heads / n_kv).
//
// BIT-EXACTNESS. The C code is compiled with FMA contraction, so its float
// operations are: dot = fma(q[d], k[d], dot) for d = 0..hd-1 (one serial chain
// per key), s = dot*scale, e = expf(s - m), sum = sum + e for j ascending (one
// serial chain per head), w = e*inv, o[d] = fma(w, v[d], o[d]) for j ascending
// (one serial chain per output element). We may do these chains in any order
// and on any thread -- but every chain itself must run exactly like C:
//   * each score is its own chain       -> keys can be split across threads freely
//   * the max is exact in any order     -> vector max, any split
//   * each e_j is independent           -> vector expf (bit-exact EXP8)
//   * the sum is ONE chain per head     -> one thread walks it, key by key
//   * o[d] is ONE chain per element d   -> threads may split the d's (slices of
//     16 floats = one 64-byte cache line of a V row), never the keys.
//
// DATA LAYOUT. K cache: blocked (cap % 8 == 0): 8 consecutive slots form a
// channel-major block, K[s][c] at k + (s>>3)*8*kv_dim + c*8 + (s&7), so one ymm
// load gives channel c of 8 keys and 8 keys are scored at once with 8
// independent FMA chains in the 8 lanes. Otherwise row-major (scalar chains).
// V cache: row-major, row of slot s at v + s*kv_dim.
// Scores of (row r, head h) live in the scratch at sc + (r*n_heads + h)*scstride,
// index i = key - glo, where glo = lo & ~7 for the blocked layout (the block
// holding key lo starts there) and glo = lo for row-major. The slack lanes
// before lo and after pos are set to -inf, so the vector max ignores them and
// their expf is exactly +0 (adding +0 to the running sum changes nothing).
//
// GQA REUSE. All `group` query heads of a kv head read the same K and V rows,
// so every pass below loads a piece of K or V once (from DRAM / L3) and uses it
// for all heads of the group while it sits in L1/L2. This cuts the cache
// traffic by a factor of `group` (7 for Qwen2.5-0.5B, 3 for SmolLM2-135M).
//
// HOW IT RUNS. Decode attention is memory-bound once the K/V of a layer are
// big: every key costs 2*kv_dim*4 bytes of DRAM traffic and only a few FMAs,
// and the i5-14400 streams ~26 GB/s. So the job is to keep enough threads
// streaming and to read every byte once. Two schedules:
//
// (1) Split decode: one query row (decode), blocked K, more than
//     att_split_min keys, a K+V span below att_split_max_bytes, >= 2 threads.
//     ONE pool dispatch; every pool thread runs att_dec_worker, which pulls
//     numbered tasks in ascending order:
//       A  scores. Task = a chunk of keys (multiple of 64), all heads; the K
//          chunk is read once, and each kv head's part is reused by its
//          `group` query heads while it sits in L1.
//       B  softmax of one head (max, expf, the serial sum, w = e*inv);
//          waits until all A tasks are done.
//       C  V of one kv head x att_vslices 16-float slices of head_dim: streams
//          those 64-byte pieces of every V row in key order and accumulates
//          all `group` heads at once (o[h][slice] += w_h[j] * v_j[slice]);
//          waits for the B tasks of its kv head.
//     Qwen (14 heads, 2 kv): 32 A + 14 B + 8 C tasks instead of 14 head tasks.
//     Runs on pool_mem_threads threads (the 6 P-cores; see att_dec_threads).
//     It beats per-head tasks at every context: 1.2-1.6x while K/V partly
//     fit in the caches, and ~1.5x at long context on 6 threads (14 or 9
//     per-head tasks do not divide evenly over 6 threads; key chunks do).
//
// (2) Row tasks: everything else (prefill batches, short or very long
//     decode). Task = up to att_rows_per_task consecutive query rows x the
//     heads of one kv group (all `group` of them; or ONE head when a single
//     row runs on the pool, so that n_heads tasks keep the threads busy).
//     Prefill uses every pool thread; a single row uses att_dec_threads.
//     The task scores K tile by tile (att_ktile keys, each tile reused by
//     every row and head of the task from L1/L2), runs the softmax of every
//     (row, head), then walks V in tiles of att_vtile keys (reused by every
//     row, head and slice from L1). For prefill this cuts the K/V cache
//     traffic by rows x group (28x for Qwen) compared with (row, head) tasks.
//
// Internal helpers (rc_init, ks_abs, ks_run, ks_rowmajor, sm_rc, softmax_n,
// sm_prep, sm_finish, zero_o, v_abs, v_tile, vacc_run) take their arguments in
// the registers listed at each one and always get rbx = X (the extended args,
// see XA_*). They keep the Win64 callee-saved GPRs but clobber ALL ymm
// registers: the pool task functions (att_row_tasks, att_dec_worker) save
// xmm6-15 once on entry.
// ===========================================================================

// ---- tuning -----------------------------------------------------------------
// Data words (not #defines) so tests/att_test.c can sweep them; the values were
// picked by measurement on the i5-14400 (tests/att_test.c bench / prefill).
#define ATT_RT_MAX 8                    // max query rows per row task (frame space)
        .data
        .p2align 3
        .globl  att_split_min, att_split_max_bytes, att_chunks_per_thread
        .globl  att_vslices, att_vpf_shift, att_kpf_lines
        .globl  att_rows_per_task, att_ktile, att_vtile
att_split_min:          .quad 64        // split decode from this many keys ...
att_split_max_bytes:    .quad 0x4000000000000000 // ... up to this K+V span (no limit: with the
                                        // 6 P-core threads split wins at every length)
att_chunks_per_thread:  .quad 2         // split decode: A tasks per pool thread
att_vslices:            .quad 1         // split decode: 16-float V slices per C task
att_vpf_shift:          .quad 5         // V prefetch distance: 2^5 = 32 rows ahead
att_kpf_lines:          .quad 2         // K prefetch: first lines of each block of the next group
att_rows_per_task:      .quad 4         // row tasks: query rows per task (1..ATT_RT_MAX)
att_ktile:              .quad 512       // row tasks: keys per K tile (multiple of 8)
att_vtile:              .quad 64        // row tasks / C tasks: keys per V tile
        .globl  att_dec_threads_n
att_dec_threads_n:      .quad 0         // decode (1 row) threads: 0 = pool_mem_threads, <0 = all
        .text

// ---- X: AttnArgs copy (0..127, the AT_* fields) + derived values, on attention()'s stack
#define XA_HD4    128   // hd*4: bytes per head in q / o
#define XA_KVD4   136   // kv_dim*4: V row stride (and the row-major K row stride)
#define XA_KBS    144   // K stride: blocked = 8*kv_dim*4 per 8-slot block, row-major = kv_dim*4
#define XA_SCS4   152   // scstride*4: bytes between the score rows of two tasks
#define XA_QROW4  160   // n_heads*hd*4: bytes per query row in q / o
#define XA_NBR    168   // cap/8: blocks in the ring (blocked layout)
#define XA_BLK    176   // 1 = blocked K layout
#define XA_NSL    184   // V slices per head: ceil(hd/16)
#define XA_R      192   // row tasks: query rows per task
#define XA_CK     200   // split decode: keys per pass-A chunk (multiple of 64)
#define XA_PB     208   // split decode: first key of chunk 0 (lo & ~7)
#define XA_HPT    216   // row tasks: query heads per task (group, or 1 for decode)
#define XA_TPR    224   // row tasks: tasks per row group (n_heads / XA_HPT)
#define XA_KT     232   // row tasks: K tile (keys)
#define XA_VT     240   // row tasks: V tile (keys)
#define XA_NCH    248   // split decode: A tasks (key chunks, all heads)
#define XA_TB     256   // split decode: B + C tasks per kv head (group + NCT)
#define XA_NTASK  264   // split decode: total tasks (NCH + TB * n_kv)
#define XA_NCT    272   // split decode: pass-C tasks per kv head
#define XA_VSL    280   // split decode: slices per pass-C task
#define XA_NEXT   320   // split decode: next task to claim (lock xadd)
#define ATT_MAXKV 32    // split decode needs n_kv <= this (counter space)
#define XA_ADONE  384                           // finished A tasks (one counter)
#define XA_BDONE  (XA_ADONE + 8 * ATT_MAXKV)    // finished B tasks, per kv head
#define XA_SIZE   (XA_BDONE + 8 * ATT_MAXKV)

// ---- RC: one query row x some heads of ONE kv group, filled by rc_init
#define RC_LO     0     // first key (window start)
#define RC_CNT    8     // keys lo..pos
#define RC_OFF    16    // score index of key lo (lo - glo)
#define RC_P      24    // padded score count (multiple of 8)
#define RC_NB     32    // blocked: score blocks covering glo..pos
#define RC_RB0    40    // blocked: ring block of score block 0
#define RC_SLOT0  48    // slot of key lo
#define RC_POS    56    // position of the query row (last key)
#define RC_Q      64    // q of the first head of the group (hd floats per head)
#define RC_SC     72    // score row of the first head (heads XA_SCS4 apart)
#define RC_O      80    // output of the first head
#define RC_KB     88    // K base of the kv head (blocked: kc + kvh*hd*8*4)
#define RC_VB     96    // V base of the kv head (vc + kvh*hd*4)
#define RC_G      104   // number of query heads (consecutive, same kv head)
#define RC_SIZE   112

// ---- pool task frame (att_row_tasks, att_dec_worker): xmm6-15, RC records, locals
#define TF_RC     160                                   // ATT_RT_MAX RCs
#define TF_L      (TF_RC + ATT_RT_MAX * RC_SIZE)        // 8 qword locals
#define TF        (TF_L + 64 + 8)                       // 8 pushes + 8 + TF = 16-aligned

// void attention(AttnArgs* a)
//   Fills a->scale and a->group (like before), copies the args into X on the
//   stack, adds the derived values, and picks split decode or row tasks.
        .globl  attention
        .p2align 4
attention:
        push    rbx
        push    rsi
        push    rdi
        push    r12
        sub     rsp, 48 + XA_SIZE + 8           // 40 + 48 + XA_SIZE + 8: 16-aligned for calls
        mov     rsi, rcx                        // the caller's AttnArgs
        lea     rbx, [rsp + 48]                 // X (rsp+0..31 = shadow space, +32 = 5th arg)
        cmp     qword ptr [rsi + AT_NROWS], 0
        jle     .Lattn_done                     // no query rows: nothing to do
        vmovups ymm0, ymmword ptr [rsi]         // copy the 128 bytes of AttnArgs
        vmovups ymm1, ymmword ptr [rsi + 32]
        vmovups ymm2, ymmword ptr [rsi + 64]
        vmovups ymm3, ymmword ptr [rsi + 96]
        vmovups ymmword ptr [rbx], ymm0
        vmovups ymmword ptr [rbx + 32], ymm1
        vmovups ymmword ptr [rbx + 64], ymm2
        vmovups ymmword ptr [rbx + 96], ymm3
        mov     rax, qword ptr [rbx + AT_NHEADS]
        cqo
        idiv    qword ptr [rbx + AT_NKV]
        mov     qword ptr [rbx + AT_GROUP], rax         // heads per kv head
        mov     qword ptr [rsi + AT_GROUP], rax
        mov     rax, qword ptr [rbx + AT_HD]
        vcvtsi2ss xmm0, xmm0, rax
        vsqrtss xmm0, xmm0, xmm0
        vmovss  xmm1, dword ptr [rip + C_ONE]
        vdivss  xmm1, xmm1, xmm0                        // scale = 1.0f / sqrtf((float)hd)
        vmovss  dword ptr [rbx + AT_SCALE], xmm1
        vmovss  dword ptr [rsi + AT_SCALE], xmm1
        // derived sizes (bytes)
        mov     rax, qword ptr [rbx + AT_HD]
        shl     rax, 2
        mov     qword ptr [rbx + XA_HD4], rax           // hd*4
        mov     rcx, rax
        imul    rcx, qword ptr [rbx + AT_NKV]
        mov     qword ptr [rbx + XA_KVD4], rcx          // kv_dim*4
        mov     rdx, rax
        imul    rdx, qword ptr [rbx + AT_NHEADS]
        mov     qword ptr [rbx + XA_QROW4], rdx         // n_heads*hd*4
        mov     rdx, qword ptr [rbx + AT_SCSTRIDE]
        shl     rdx, 2
        mov     qword ptr [rbx + XA_SCS4], rdx
        add     rax, 63
        shr     rax, 6
        mov     qword ptr [rbx + XA_NSL], rax           // ceil(hd/16) V slices
        mov     rax, qword ptr [rbx + AT_CAP]
        mov     rdx, rax
        shr     rdx, 3
        mov     qword ptr [rbx + XA_NBR], rdx           // cap/8
        xor     edx, edx
        test    eax, 7
        sete    dl                                      // blocked iff cap % 8 == 0
        mov     qword ptr [rbx + XA_BLK], rdx
        test    edx, edx
        jz      1f
        shl     rcx, 3                                  // blocked: 8 slots per K block
1:      mov     qword ptr [rbx + XA_KBS], rcx

        mov     rax, qword ptr [rip + att_ktile]
        mov     qword ptr [rbx + XA_KT], rax
        mov     rax, qword ptr [rip + att_vtile]
        mov     qword ptr [rbx + XA_VT], rax

        // ---- split decode? one row, blocked K, enough keys, a pool ----
        cmp     qword ptr [rbx + AT_NROWS], 1
        jne     .Lattn_rows
        cmp     qword ptr [rbx + XA_BLK], 0
        je      .Lattn_rows
        mov     rax, qword ptr [rbx + AT_POS0]          // pos
        lea     r8, [rax + 1]                           // cnt = pos+1 ...
        mov     rdx, qword ptr [rbx + AT_WINDOW]
        test    rdx, rdx
        jle     2f
        cmp     r8, rdx
        jbe     2f
        mov     r8, rdx                                 // ... or window, if smaller
2:      cmp     r8, qword ptr [rip + att_split_min]
        jb      .Lattn_rows
        mov     rax, r8
        imul    rax, qword ptr [rbx + XA_KVD4]
        add     rax, rax                                // bytes of K + V to stream
        cmp     rax, qword ptr [rip + att_split_max_bytes]
        jae     .Lattn_rows                             // DRAM-bound: per-head tasks are as fast
        mov     rdi, r8                                 // cnt
        call    att_dec_threads
        cmp     rax, 2
        jb      .Lattn_rows
        mov     r12, rax                                // threads
        cmp     qword ptr [rbx + AT_NKV], ATT_MAXKV
        ja      .Lattn_rows
        mov     rcx, qword ptr [rbx + AT_POS0]
        inc     rcx                                     // pos+1
        mov     rsi, rcx
        sub     rcx, rdi                                // lo = pos+1-cnt
        and     rcx, -8                                 // glo: first key of chunk 0
        mov     qword ptr [rbx + XA_PB], rcx
        sub     rsi, rcx                                // keys to score (incl. slack)
        // pass-A chunks ~ threads * att_chunks_per_thread,
        // chunk = roundup64(ceil(keys / that))
        mov     rcx, r12
        imul    rcx, qword ptr [rip + att_chunks_per_thread]
        lea     rax, [rsi + rcx - 1]
        xor     edx, edx
        div     rcx
        add     rax, 63
        and     rax, -64
        mov     qword ptr [rbx + XA_CK], rax
        mov     rcx, rax
        lea     rax, [rsi + rcx - 1]
        xor     edx, edx
        div     rcx
        mov     qword ptr [rbx + XA_NCH], rax           // A tasks (key chunks)
        mov     r8, rax
        mov     rcx, qword ptr [rip + att_vslices]      // slices per C task
        mov     qword ptr [rbx + XA_VSL], rcx
        mov     rax, qword ptr [rbx + XA_NSL]
        lea     rax, [rax + rcx - 1]
        xor     edx, edx
        div     rcx
        mov     qword ptr [rbx + XA_NCT], rax           // C tasks per kv head
        add     rax, qword ptr [rbx + AT_GROUP]
        mov     qword ptr [rbx + XA_TB], rax            // B + C tasks per kv head
        imul    rax, qword ptr [rbx + AT_NKV]
        add     rax, r8                                 // + the A tasks
        mov     qword ptr [rbx + XA_NTASK], rax
        xor     eax, eax                                // clear the task counters
        mov     qword ptr [rbx + XA_NEXT], rax
        xor     ecx, ecx
7:      mov     qword ptr [rbx + XA_ADONE + rcx*8], rax
        mov     qword ptr [rbx + XA_BDONE + rcx*8], rax
        inc     rcx
        cmp     rcx, qword ptr [rbx + AT_NKV]
        jb      7b
        lea     rcx, [rip + att_dec_worker]             // one worker per decode thread
        mov     rdx, rbx
        mov     r8, r12
        mov     r9d, 1
        mov     qword ptr [rsp + 32], r12               // pool_run_n(fn, X, n, 1, nthreads)
        call    pool_run_n
        jmp     .Lattn_done

        // ---- row tasks: (row group, heads) ----
        // Normally a task = R rows x all `group` heads of one kv head, so each
        // K/V tile is reused R*group times (prefill; and inline runs).
        // A single query row run on the pool (short decode below the split
        // threshold) uses one task per head instead: n_heads tasks keep more
        // threads busy than n_kv tasks, and the few keys sit in L2/L3 anyway.
.Lattn_rows:
        mov     rax, qword ptr [rbx + AT_NROWS]
        mov     rcx, qword ptr [rip + att_rows_per_task]
        mov     edx, ATT_RT_MAX
        cmp     rcx, rdx
        jbe     2f
        mov     rcx, rdx                                // at most ATT_RT_MAX RCs fit the frame
2:      cmp     rax, rcx
        jae     3f
        mov     rcx, rax                                // fewer rows than a group
3:      mov     qword ptr [rbx + XA_R], rcx
        lea     rax, [rax + rcx - 1]
        xor     edx, edx
        div     rcx
        mov     r12, rax                                // row groups
        xor     esi, esi                                // rsi = 1: run on the pool
        mov     rax, qword ptr [rbx + AT_POS0]
        add     rax, qword ptr [rbx + AT_NROWS]         // ~ keys per row (upper bound)
        imul    rax, qword ptr [rbx + AT_NROWS]
        imul    rax, qword ptr [rbx + AT_NHEADS]
        imul    rax, qword ptr [rbx + AT_HD]            // work estimate
        cmp     rax, ATTN_THREAD_MIN
        jb      4f
        mov     esi, 1
4:      mov     rdx, qword ptr [rbx + AT_GROUP]         // heads per task: group ...
        cmp     qword ptr [rbx + AT_NROWS], 1
        jne     5f
        test    esi, esi
        jz      5f
        mov     edx, 1                                  // ... or 1 (one row, pooled)
5:      mov     qword ptr [rbx + XA_HPT], rdx
        mov     r8, rdx
        mov     rax, qword ptr [rbx + AT_NHEADS]
        xor     edx, edx
        div     r8
        mov     qword ptr [rbx + XA_TPR], rax           // tasks per row group
        imul    r12, rax                                // tasks
        cmp     r12, 2
        jb      6f
        test    esi, esi
        jz      6f
        xor     eax, eax                                // prefill: all pool threads
        cmp     qword ptr [rbx + AT_NROWS], 1
        jne     7f
        call    att_dec_threads                         // decode: the memory threads
7:      lea     rcx, [rip + att_row_tasks]              // pool_run_n(fn, X, tasks, 1, nthreads)
        mov     rdx, rbx
        mov     r8, r12
        mov     r9d, 1
        mov     qword ptr [rsp + 32], rax
        call    pool_run_n
        jmp     .Lattn_done
6:      mov     rcx, rbx                                // small: run inline
        xor     edx, edx
        mov     r8, r12
        call    att_row_tasks
.Lattn_done:
        add     rsp, 48 + XA_SIZE + 8
        pop     r12
        pop     rdi
        pop     rsi
        pop     rbx
        vzeroupper
        ret

// att_dec_threads: rax = threads for single-row (decode) attention.
//   in: rbx = X (reads AT_NHEADS). Clobbers the volatile registers.
// att_dec_threads_n > 0: that many (clipped to pool_threads()); < 0: all.
// 0 (default): pool_mem_threads (the 6 P-cores: waking the sleeping
// E-core/HT threads every token costs more than it gains, and their spinning
// slows the matvecs that follow) -- unless n_heads > 2*pool_mem_threads, then
// all threads: the softmax is one latency-bound task per head, and with more
// than two rounds of them the extra threads win. Measured end to end at a
// 4258-token prompt: Qwen (14 heads) 34.7 tok/s on 16 threads vs 30.4 on 6;
// SmolLM (9 heads) 50.0 on 6 vs 43.2 on 16.
        .p2align 4
att_dec_threads:
        push    rbx
        sub     rsp, 32
        call    pool_threads
        mov     rcx, qword ptr [rip + att_dec_threads_n]
        test    rcx, rcx
        js      2f                              // < 0: all
        jnz     1f
        mov     rcx, qword ptr [rip + pool_mem_threads]
        test    rcx, rcx
        jle     2f                              // pool not sized yet: all
        lea     rdx, [rcx + rcx]
        cmp     qword ptr [rbx + AT_NHEADS], rdx
        ja      2f                              // > 2 rounds of softmax tasks: all
1:      cmp     rcx, rax
        jae     2f
        mov     rax, rcx
2:      add     rsp, 32
        pop     rbx
        ret

// ---------------------------------------------------------------------------
// Pool task wrappers. fn(X, t0, t1) runs tasks [t0, t1). Win64 entry: they save
// every callee-saved register (incl. xmm6-15), set rbx = X, r12 = t, r13 = t1.
// ---------------------------------------------------------------------------
.macro TASK_ENTER
        push    rbx
        push    rbp
        push    rsi
        push    rdi
        push    r12
        push    r13
        push    r14
        push    r15
        sub     rsp, TF
        vmovaps xmmword ptr [rsp], xmm6
        vmovaps xmmword ptr [rsp + 16], xmm7
        vmovaps xmmword ptr [rsp + 32], xmm8
        vmovaps xmmword ptr [rsp + 48], xmm9
        vmovaps xmmword ptr [rsp + 64], xmm10
        vmovaps xmmword ptr [rsp + 80], xmm11
        vmovaps xmmword ptr [rsp + 96], xmm12
        vmovaps xmmword ptr [rsp + 112], xmm13
        vmovaps xmmword ptr [rsp + 128], xmm14
        vmovaps xmmword ptr [rsp + 144], xmm15
        mov     rbx, rcx
        mov     r12, rdx
        mov     r13, r8
.endm
.macro TASK_LEAVE
        vmovaps xmm6, xmmword ptr [rsp]
        vmovaps xmm7, xmmword ptr [rsp + 16]
        vmovaps xmm8, xmmword ptr [rsp + 32]
        vmovaps xmm9, xmmword ptr [rsp + 48]
        vmovaps xmm10, xmmword ptr [rsp + 64]
        vmovaps xmm11, xmmword ptr [rsp + 80]
        vmovaps xmm12, xmmword ptr [rsp + 96]
        vmovaps xmm13, xmmword ptr [rsp + 112]
        vmovaps xmm14, xmmword ptr [rsp + 128]
        vmovaps xmm15, xmmword ptr [rsp + 144]
        add     rsp, TF
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
.endm

// att_row_tasks: task t = rowgroup*TPR + j: rows rowgroup*R .. +R-1, query
// heads j*HPT .. j*HPT+HPT-1 (all inside one kv group).
//   locals: [TF_L] rows in this task, [TF_L+8] first key (8-aligned),
//           [TF_L+16] last key + 1 (over all rows of the task)
        .p2align 4
att_row_tasks:
        TASK_ENTER
        cmp     r12, r13
        jae     .Lrt_ret
.Lrt_task:
        mov     rax, r12
        xor     edx, edx
        div     qword ptr [rbx + XA_TPR]        // rax = row group, rdx = head block
        mov     r15, rdx
        imul    r15, qword ptr [rbx + XA_HPT]   // first head
        mov     r14, rax
        imul    r14, qword ptr [rbx + XA_R]     // first row
        mov     rsi, qword ptr [rbx + XA_R]
        mov     rax, qword ptr [rbx + AT_NROWS]
        sub     rax, r14
        cmp     rsi, rax
        jbe     1f
        mov     rsi, rax                        // last group may be short
1:      mov     qword ptr [rsp + TF_L], rsi     // rows in this task
        xor     edi, edi
2:      imul    rcx, rdi, RC_SIZE               // RC of row i: geometry + pointers
        lea     rcx, [rsp + TF_RC + rcx]
        lea     rdx, [r14 + rdi]
        mov     r8, r15
        mov     r9, qword ptr [rbx + XA_HPT]
        call    rc_init
        inc     rdi
        cmp     rdi, rsi
        jb      2b
        mov     rax, qword ptr [rsp + TF_RC + RC_LO]    // lo of the first row is the smallest
        and     rax, -8
        mov     qword ptr [rsp + TF_L + 8], rax
        lea     rcx, [rsi - 1]
        imul    rcx, rcx, RC_SIZE
        mov     rax, qword ptr [rsp + TF_RC + RC_POS + rcx]   // pos of the last row
        inc     rax
        mov     qword ptr [rsp + TF_L + 16], rax

        // scores, K tile by K tile (each tile serves every row and head)
        mov     rbp, qword ptr [rsp + TF_L + 8]
3:      lea     rsi, [rsp + TF_RC]
        mov     rdi, qword ptr [rsp + TF_L]
4:      mov     rcx, rsi
        mov     rdx, rbp
        mov     r8, rbp
        add     r8, qword ptr [rbx + XA_KT]
        call    ks_abs
        add     rsi, RC_SIZE
        dec     rdi
        jnz     4b
        add     rbp, qword ptr [rbx + XA_KT]
        cmp     rbp, qword ptr [rsp + TF_L + 16]
        jb      3b

        // softmax of every (row, head); clear the outputs
        lea     rsi, [rsp + TF_RC]
        mov     rdi, qword ptr [rsp + TF_L]
5:      mov     rcx, rsi
        call    sm_rc
        mov     rcx, qword ptr [rsi + RC_O]
        xor     edx, edx
        mov     r8, qword ptr [rbx + XA_NSL]
        mov     r9, qword ptr [rsi + RC_G]
        call    zero_o
        add     rsi, RC_SIZE
        dec     rdi
        jnz     5b

        // o += w * V, V tile by V tile (each tile serves every row, head, slice)
        mov     rbp, qword ptr [rsp + TF_L + 8]
6:      lea     rsi, [rsp + TF_RC]
        mov     rdi, qword ptr [rsp + TF_L]
7:      mov     rcx, rsi
        mov     rdx, rbp
        mov     r8, rbp
        add     r8, qword ptr [rbx + XA_VT]
        xor     r9d, r9d
        mov     r10, qword ptr [rbx + XA_NSL]
        call    v_abs
        add     rsi, RC_SIZE
        dec     rdi
        jnz     7b
        add     rbp, qword ptr [rbx + XA_VT]
        cmp     rbp, qword ptr [rsp + TF_L + 16]
        jb      6b

        inc     r12
        cmp     r12, r13
        jb      .Lrt_task
.Lrt_ret:
        TASK_LEAVE

// att_dec_worker(X, i0, i1): split decode. A pool fn that ignores its chunk
// and pulls tasks from X instead, in ascending order (lock xadd on XA_NEXT):
//   t <  NCH   A: scores of every head for key chunk t (kv group by kv group:
//              the chunk of K is read once, contiguously, and each kv head's
//              part of a block is reused by the `group` heads that share it)
//   then, kv head by kv head (kvh = (t-NCH) / TB, r = (t-NCH) % TB):
//   r < group  B: softmax of head kvh*group + r; waits until all A are done
//   else       C: V slices of the group; waits for the group's B tasks
// A task only ever waits for tasks with smaller numbers. Those were claimed
// earlier by workers that are running them right now (claim and run happen in
// the same loop), so the waits always end -- whatever order or thread count
// the pool uses. With this order the latency-bound softmax of one kv group
// overlaps the memory-bound V streaming of the previous one.
        .p2align 4
att_dec_worker:
        TASK_ENTER
.Ldw_next:
        mov     eax, 1
        lock xadd qword ptr [rbx + XA_NEXT], rax        // t = next++
        cmp     rax, qword ptr [rbx + XA_NTASK]
        jae     .Ldw_ret
        sub     rax, qword ptr [rbx + XA_NCH]
        jae     .Ldw_bc
        // ---- A: scores of key chunk t, all kv heads ----
        add     rax, qword ptr [rbx + XA_NCH]
        imul    rax, qword ptr [rbx + XA_CK]
        add     rax, qword ptr [rbx + XA_PB]
        mov     rbp, rax                                // first key of the chunk
        xor     r12d, r12d                              // h0 of the kv group
1:      lea     rcx, [rsp + TF_RC]
        xor     edx, edx                                // row 0
        mov     r8, r12
        mov     r9, qword ptr [rbx + AT_GROUP]
        call    rc_init
        lea     rcx, [rsp + TF_RC]
        mov     rdx, rbp
        mov     r8, rbp
        add     r8, qword ptr [rbx + XA_CK]
        call    ks_abs
        add     r12, qword ptr [rbx + AT_GROUP]
        cmp     r12, qword ptr [rbx + AT_NHEADS]
        jb      1b
        lock inc qword ptr [rbx + XA_ADONE]
        jmp     .Ldw_next
.Ldw_bc:
        xor     edx, edx
        div     qword ptr [rbx + XA_TB]
        mov     r15, rax                                // kv head
        mov     r14, rdx                                // task within the kv head
        mov     r12, rax
        imul    r12, qword ptr [rbx + AT_GROUP]         // h0
        cmp     r14, qword ptr [rbx + AT_GROUP]
        jae     .Ldw_c
        // ---- B: softmax of head h0 + r14 ----
        mov     rax, qword ptr [rbx + XA_NCH]
1:      cmp     qword ptr [rbx + XA_ADONE], rax         // every chunk scored?
        jae     2f
        pause
        jmp     1b
2:      lea     rcx, [rsp + TF_RC]
        xor     edx, edx
        lea     r8, [r12 + r14]
        mov     r9d, 1
        call    rc_init
        lea     rcx, [rsp + TF_RC]
        call    sm_rc
        lock inc qword ptr [rbx + XA_BDONE + r15*8]
        jmp     .Ldw_next
.Ldw_c: sub     r14, qword ptr [rbx + AT_GROUP]
        imul    r14, qword ptr [rbx + XA_VSL]           // first V slice of this task
        mov     rax, qword ptr [rbx + AT_GROUP]
3:      cmp     qword ptr [rbx + XA_BDONE + r15*8], rax // every head's weights ready?
        jae     4f
        pause
        jmp     3b
4:      lea     rcx, [rsp + TF_RC]
        xor     edx, edx
        mov     r8, r12
        mov     r9, qword ptr [rbx + AT_GROUP]
        call    rc_init
        mov     r13, r14
        add     r13, qword ptr [rbx + XA_VSL]           // slices [r14, r13)
        mov     rcx, qword ptr [rsp + TF_RC + RC_O]
        mov     rdx, r14
        mov     r8, r13
        mov     r9, qword ptr [rbx + AT_GROUP]
        call    zero_o                                  // o[g][slices] = 0
        mov     rbp, qword ptr [rsp + TF_RC + RC_LO]    // keys lo..pos, V tile by V tile
5:      lea     rcx, [rsp + TF_RC]
        mov     rdx, rbp
        mov     r8, rbp
        add     r8, qword ptr [rbx + XA_VT]
        mov     r9, r14
        mov     r10, r13
        call    v_abs
        add     rbp, qword ptr [rbx + XA_VT]
        cmp     rbp, qword ptr [rsp + TF_RC + RC_POS]
        jbe     5b
        jmp     .Ldw_next
.Ldw_ret:
        TASK_LEAVE

// ---------------------------------------------------------------------------
// rc_init: fill RC for (query row, heads h0 .. h0+G-1 of one kv group)
//   in: rbx = X, rcx = RC, rdx = row, r8 = h0, r9 = G ; clobbers rax rdx r8-r11
// ---------------------------------------------------------------------------
        .p2align 4
rc_init:
        mov     qword ptr [rcx + RC_G], r9
        mov     r10, rdx                        // row
        mov     r11, r8                         // h0
        mov     rax, rdx
        add     rax, qword ptr [rbx + AT_POS0]  // pos
        mov     qword ptr [rcx + RC_POS], rax
        xor     r8d, r8d                        // lo = 0 ...
        mov     rdx, qword ptr [rbx + AT_WINDOW]
        test    rdx, rdx
        jle     1f
        mov     r9, rax
        sub     r9, rdx
        inc     r9                              // pos - window + 1
        test    r9, r9
        jle     1f
        mov     r8, r9                          // ... or the window start
1:      mov     qword ptr [rcx + RC_LO], r8
        sub     rax, r8
        inc     rax
        mov     qword ptr [rcx + RC_CNT], rax   // cnt = pos - lo + 1
        mov     r9, rax
        mov     rax, r8
        xor     edx, edx
        div     qword ptr [rbx + AT_CAP]
        mov     qword ptr [rcx + RC_SLOT0], rdx // slot of key lo
        cmp     qword ptr [rbx + XA_BLK], 0
        je      2f
        shr     rdx, 3
        mov     qword ptr [rcx + RC_RB0], rdx   // ring block of glo (same block as lo)
        mov     rax, r8
        and     eax, 7
        mov     qword ptr [rcx + RC_OFF], rax   // lo - glo
        lea     rax, [rax + r9 + 7]
        shr     rax, 3
        mov     qword ptr [rcx + RC_NB], rax    // blocks covering glo..pos
        shl     rax, 3
        mov     qword ptr [rcx + RC_P], rax
        jmp     3f
2:      mov     qword ptr [rcx + RC_OFF], 0     // row-major: scores start at key lo
        mov     qword ptr [rcx + RC_NB], 0
        mov     qword ptr [rcx + RC_RB0], 0
        lea     rax, [r9 + 7]
        and     rax, -8
        mov     qword ptr [rcx + RC_P], rax
3:      mov     r8, r10
        imul    r8, qword ptr [rbx + AT_NHEADS]
        add     r8, r11
        imul    r8, qword ptr [rbx + XA_SCS4]
        add     r8, qword ptr [rbx + AT_SC]
        mov     qword ptr [rcx + RC_SC], r8     // sc + (row*n_heads + h0)*scstride
        mov     rax, r11
        imul    rax, qword ptr [rbx + XA_HD4]
        mov     r8, r10
        imul    r8, qword ptr [rbx + XA_QROW4]
        add     r8, rax                         // row*n_heads*hd*4 + h0*hd*4
        mov     r9, qword ptr [rbx + AT_Q]
        add     r9, r8
        mov     qword ptr [rcx + RC_Q], r9
        add     r8, qword ptr [rbx + AT_O]
        mov     qword ptr [rcx + RC_O], r8
        mov     rax, r11
        xor     edx, edx
        div     qword ptr [rbx + AT_GROUP]      // kv head = h0 / group
        imul    rax, qword ptr [rbx + XA_HD4]   // kvh*hd*4
        mov     r8, qword ptr [rbx + AT_VC]
        add     r8, rax
        mov     qword ptr [rcx + RC_VB], r8
        cmp     qword ptr [rbx + XA_BLK], 0
        je      4f
        shl     rax, 3                          // blocked: a channel row is 8 floats
4:      add     rax, qword ptr [rbx + AT_KC]
        mov     qword ptr [rcx + RC_KB], rax
        ret

// ---------------------------------------------------------------------------
// ks_abs: scores of every head of RC's group for the keys of [P0, P1) that the
// row attends to (P0, P1 multiples of 8; clipped to lo..pos).
//   in: rbx = X, rcx = RC, rdx = P0, r8 = P1 ; preserves callee-saved GPRs
// ---------------------------------------------------------------------------
        .p2align 4
ks_abs:
        push    rsi
        push    rdi
        push    rbp
        mov     rsi, rcx
        cmp     qword ptr [rbx + XA_BLK], 0
        je      .Lka_rm
        // blocked: work in absolute block numbers (block b = keys 8b..8b+7)
        shr     rdx, 3
        shr     r8, 3
        mov     rax, qword ptr [rsi + RC_LO]
        shr     rax, 3                          // block of glo = score block 0
        mov     r9, qword ptr [rsi + RC_POS]
        shr     r9, 3
        inc     r9                              // one past the block of pos
        cmp     rdx, rax
        jae     1f
        mov     rdx, rax                        // a = max(P0/8, first block)
1:      cmp     r8, r9
        jbe     2f
        mov     r8, r9                          // b = min(P1/8, last block + 1)
2:      cmp     rdx, r8
        jae     .Lka_ret
        mov     rdi, rdx
        sub     rdi, rax                        // i = score block of a
        mov     rbp, r8
        sub     rbp, rdx                        // blocks to do
        mov     r8, qword ptr [rsi + RC_RB0]
        add     r8, rdi                         // ring block of a ...
        cmp     r8, qword ptr [rbx + XA_NBR]
        jb      3f
        sub     r8, qword ptr [rbx + XA_NBR]    // ... modulo cap/8
3:      mov     r9, qword ptr [rbx + XA_NBR]
        sub     r9, r8                          // blocks left before the ring wraps
        cmp     r9, rbp
        jbe     4f
        mov     r9, rbp
4:      sub     rbp, r9                         // blocks after the wrap (usually 0)
        mov     rcx, qword ptr [rsi + RC_Q]
        mov     rdx, rdi
        shl     rdx, 5
        add     rdx, qword ptr [rsi + RC_SC]    // &score[h0][i*8]
        add     rdi, r9                         // score block after the wrap
        mov     r10, qword ptr [rsi + RC_G]
        mov     r11, qword ptr [rsi + RC_KB]
        call    ks_run
        test    rbp, rbp
        jz      .Lka_ret
        mov     rcx, qword ptr [rsi + RC_Q]
        mov     rdx, rdi
        shl     rdx, 5
        add     rdx, qword ptr [rsi + RC_SC]
        xor     r8d, r8d                        // ring block 0
        mov     r9, rbp
        mov     r10, qword ptr [rsi + RC_G]
        mov     r11, qword ptr [rsi + RC_KB]
        call    ks_run
        jmp     .Lka_ret
.Lka_rm:                                        // row-major K: absolute keys
        mov     rax, qword ptr [rsi + RC_LO]
        mov     r9, qword ptr [rsi + RC_POS]
        inc     r9
        cmp     rdx, rax
        jae     5f
        mov     rdx, rax
5:      cmp     r8, r9
        jbe     6f
        mov     r8, r9
6:      cmp     rdx, r8
        jae     .Lka_ret
        mov     rdi, rdx
        sub     rdi, rax                        // score index of a (= a - lo)
        mov     r9, r8
        sub     r9, rdx                         // keys
        mov     r8, qword ptr [rsi + RC_SLOT0]
        add     r8, rdi
        cmp     r8, qword ptr [rbx + AT_CAP]
        jb      7f
        sub     r8, qword ptr [rbx + AT_CAP]    // slot of a
7:      mov     rcx, qword ptr [rsi + RC_Q]
        lea     rdx, [rdi*4]
        add     rdx, qword ptr [rsi + RC_SC]
        mov     r10, qword ptr [rsi + RC_G]
        mov     r11, qword ptr [rsi + RC_KB]
        call    ks_rowmajor
.Lka_ret:
        pop     rbp
        pop     rdi
        pop     rsi
        ret

// ---------------------------------------------------------------------------
// ks_run: scores of G query heads (that share one kv head) for nb K blocks that
// are contiguous in the ring. 8 blocks (64 keys) at a time: for each head, 8
// accumulators = 64 independent serial FMA chains (enough to hide the FMA
// latency); the 8 blocks (2 KB each at hd 64) stay in L1 for the next head.
//   in: rbx = X, rcx = q of head 0, rdx = &score[head 0][i0*8], r8 = ring block
//       of i0, r9 = nb, r10 = G, r11 = K base of the kv head
//   preserves callee-saved GPRs; clobbers rax rcx rdx r8-r11, ymm0, ymm7-15
// ---------------------------------------------------------------------------
// KS_GROUP N: N blocks (8, 4, 2 or 1) for every head. Registers:
//   rsi = q of head 0, rdi = score ptr of head 0 at this group, r12 = ring block,
//   r14 = G, r15 = K base, rbp = block stride ; r11 walks q, rcx = q end,
//   rax / r9 / r10 walk blocks 0 / 3 / 5 channel by channel (+32 bytes per d).
.macro KS_GROUP N
        mov     r11, rsi                        // q of head 0
        mov     rdx, rdi                        // scores of head 0
        mov     r8, r14                         // heads left
91:     mov     rcx, r11
        add     rcx, qword ptr [rbx + XA_HD4]   // end of this head's q
        mov     rax, r12
        imul    rax, rbp
        add     rax, r15                        // block 0 of the group
.if \N >= 4
        lea     r9, [rax + rbp*2]
        add     r9, rbp                         // block 3
.endif
.if \N == 8
        lea     r10, [r9 + rbp*2]               // block 5
.endif
        vxorps  ymm8, ymm8, ymm8
.if \N >= 2
        vxorps  ymm9, ymm9, ymm9
.endif
.if \N >= 4
        vxorps  ymm10, ymm10, ymm10
        vxorps  ymm11, ymm11, ymm11
.endif
.if \N == 8
        vxorps  ymm12, ymm12, ymm12
        vxorps  ymm13, ymm13, ymm13
        vxorps  ymm14, ymm14, ymm14
        vxorps  ymm15, ymm15, ymm15
.endif
92:     vbroadcastss ymm0, dword ptr [r11]              // q[d] in all lanes
        vfmadd231ps ymm8, ymm0, ymmword ptr [rax]       // dot = fma(q[d], k[d], dot), 8 keys
.if \N >= 2
        vfmadd231ps ymm9, ymm0, ymmword ptr [rax + rbp]
.endif
.if \N >= 4
        vfmadd231ps ymm10, ymm0, ymmword ptr [rax + rbp*2]
        vfmadd231ps ymm11, ymm0, ymmword ptr [r9]
.endif
.if \N == 8
        vfmadd231ps ymm12, ymm0, ymmword ptr [r9 + rbp]
        vfmadd231ps ymm13, ymm0, ymmword ptr [r10]
        vfmadd231ps ymm14, ymm0, ymmword ptr [r10 + rbp]
        vfmadd231ps ymm15, ymm0, ymmword ptr [r10 + rbp*2]
.endif
        add     r11, 4                          // next d
        add     rax, 32                         // next channel row of each block
.if \N >= 4
        add     r9, 32
.endif
.if \N == 8
        add     r10, 32
.endif
        cmp     r11, rcx
        jb      92b
        vmulps  ymm8, ymm8, ymm7                // s = dot * scale
        vmovups ymmword ptr [rdx], ymm8
.if \N >= 2
        vmulps  ymm9, ymm9, ymm7
        vmovups ymmword ptr [rdx + 32], ymm9
.endif
.if \N >= 4
        vmulps  ymm10, ymm10, ymm7
        vmovups ymmword ptr [rdx + 64], ymm10
        vmulps  ymm11, ymm11, ymm7
        vmovups ymmword ptr [rdx + 96], ymm11
.endif
.if \N == 8
        vmulps  ymm12, ymm12, ymm7
        vmovups ymmword ptr [rdx + 128], ymm12
        vmulps  ymm13, ymm13, ymm7
        vmovups ymmword ptr [rdx + 160], ymm13
        vmulps  ymm14, ymm14, ymm7
        vmovups ymmword ptr [rdx + 192], ymm14
        vmulps  ymm15, ymm15, ymm7
        vmovups ymmword ptr [rdx + 224], ymm15
.endif
        add     rdx, qword ptr [rbx + XA_SCS4]  // next head's scores
        dec     r8                              // (r11 = q end = next head's q)
        jnz     91b
        add     rdi, \N * 32                    // next group of blocks
        add     r12, \N
        sub     r13, \N
.endm

        .p2align 4
ks_run:
        push    rbp
        push    rsi
        push    rdi
        push    r12
        push    r13
        push    r14
        push    r15
        mov     rsi, rcx
        mov     rdi, rdx
        mov     r12, r8
        mov     r13, r9
        mov     r14, r10
        mov     r15, r11
        mov     rbp, qword ptr [rbx + XA_KBS]
        vbroadcastss ymm7, dword ptr [rbx + AT_SCALE]
.Lks_8: cmp     r13, 8
        jb      .Lks_4
        mov     rcx, qword ptr [rip + att_kpf_lines]
        test    rcx, rcx
        jz      3f
        cmp     r13, 16                         // is there a next group in this run?
        jb      3f
        lea     rax, [r12 + 8]
        imul    rax, rbp
        add     rax, r15                        // next group, block 0 (this kv head)
        xor     edx, edx
1:      xor     r8d, r8d
        mov     r9, rcx
2:      prefetcht0 byte ptr [rax + r8]
        add     r8, 64
        dec     r9
        jnz     2b
        add     rax, rbp
        inc     edx
        cmp     edx, 8
        jb      1b
3:      KS_GROUP 8
        jmp     .Lks_8
.Lks_4: cmp     r13, 4                          // tail: 4 + 2 + 1 blocks
        jb      .Lks_2
        KS_GROUP 4
.Lks_2: cmp     r13, 2
        jb      .Lks_1
        KS_GROUP 2
.Lks_1: test    r13, r13
        jz      .Lks_end
        KS_GROUP 1
.Lks_end:
        pop     r15
        pop     r14
        pop     r13
        pop     r12
        pop     rdi
        pop     rsi
        pop     rbp
        ret

// ---------------------------------------------------------------------------
// ks_rowmajor: row-major K (cap % 8 != 0): one scalar serial dot per key/head.
//   in: rbx = X, rcx = q of head 0, rdx = &score[head 0][first key],
//       r8 = slot of the first key, r9 = keys, r10 = G, r11 = K base (kc + kvh*hd*4)
//   preserves callee-saved GPRs; clobbers rax rcx rdx r8-r11, xmm0, xmm7, xmm8
// ---------------------------------------------------------------------------
        .p2align 4
ks_rowmajor:
        push    rbp
        push    rsi
        push    rdi
        push    r12
        push    r13
        push    r14
        push    r15
        mov     rsi, rcx                        // q of the current head
        mov     rdi, rdx                        // scores of the current head
        mov     r12, r8
        mov     r13, r9
        mov     r14, r10
        mov     r15, r11
        mov     rbp, qword ptr [rbx + XA_KVD4]  // row stride
        vmovss  xmm7, dword ptr [rbx + AT_SCALE]
.Lkr_head:
        mov     rcx, rsi
        add     rcx, qword ptr [rbx + XA_HD4]   // q end
        mov     r9, r12                         // slot
        xor     r10d, r10d                      // key index
1:      mov     rax, r9
        imul    rax, rbp
        add     rax, r15                        // k row
        vxorps  xmm8, xmm8, xmm8
        mov     r8, rsi
2:      vmovss  xmm0, dword ptr [r8]
        vfmadd231ss xmm8, xmm0, dword ptr [rax]         // dot = fma(q[d], k[d], dot)
        add     r8, 4
        add     rax, 4
        cmp     r8, rcx
        jb      2b
        vmulss  xmm8, xmm8, xmm7
        vmovss  dword ptr [rdi + r10*4], xmm8
        inc     r10
        inc     r9
        cmp     r9, qword ptr [rbx + AT_CAP]
        jb      3f
        xor     r9d, r9d                        // ring wrap
3:      cmp     r10, r13
        jb      1b
        mov     rsi, rcx                        // next head's q
        add     rdi, qword ptr [rbx + XA_SCS4]
        dec     r14
        jnz     .Lkr_head
        pop     r15
        pop     r14
        pop     r13
        pop     r12
        pop     rdi
        pop     rsi
        pop     rbp
        ret

// ---------------------------------------------------------------------------
// Softmax: scores -> weights w_j = expf(s_j - m) * (1/sum), in place.
//
// The sum is ONE serial chain per head in key order, as in C: 4 cycles of
// vaddss latency per key, the slowest part of the softmax. Different heads are
// independent chains, so softmax_n runs up to 4 heads (same row => same key
// range) side by side: 4 chains in flight make it ~4x faster than one at a
// time, while each chain still adds its own keys in exactly C's order.
// The leading/trailing slack lanes are -inf -> their expf is +0, and
// sum + 0 == sum exactly, so the chains simply run over all P lanes.
// ---------------------------------------------------------------------------

// sm_prep: -inf into the slack lanes of one score row, return its max.
//   in: rcx = score row (index 0 = glo), rdx = RC (RC_OFF/CNT/P)
//   out: ymm0 = max in all lanes ; clobbers rax, ymm1
        .p2align 4
sm_prep:
        vbroadcastss ymm0, dword ptr [rip + C_NEGINF]
        xor     eax, eax
1:      cmp     rax, qword ptr [rdx + RC_OFF]   // -inf into [0, off)
        jae     2f
        vmovss  dword ptr [rcx + rax*4], xmm0
        inc     rax
        jmp     1b
2:      mov     rax, qword ptr [rdx + RC_OFF]   // and into [off+cnt, P)
        add     rax, qword ptr [rdx + RC_CNT]
3:      cmp     rax, qword ptr [rdx + RC_P]
        jae     4f
        vmovss  dword ptr [rcx + rax*4], xmm0
        inc     rax
        jmp     3b
4:      vmovaps ymm1, ymm0                      // max over all lanes (order-free: exact)
        xor     eax, eax
5:      vmaxps  ymm1, ymm1, ymmword ptr [rcx + rax*4]
        add     rax, 8
        cmp     rax, qword ptr [rdx + RC_P]
        jb      5b
        vextractf128 xmm0, ymm1, 1              // horizontal max of the 8 lanes
        vmaxps  xmm0, xmm0, xmm1
        vpermilps xmm1, xmm0, 0x4E
        vmaxps  xmm0, xmm0, xmm1
        vpermilps xmm1, xmm0, 0xB1
        vmaxps  xmm0, xmm0, xmm1
        vbroadcastss ymm0, xmm0
        ret

// sm_finish: w = e * (1/sum) over the row.
//   in: rcx = row, rdx = RC, xmm0 = sum ; clobbers rax, ymm0, ymm1
        .p2align 4
sm_finish:
        vmovss  xmm1, dword ptr [rip + C_ONE]
        vdivss  xmm1, xmm1, xmm0                // inv = 1/sum
        vbroadcastss ymm1, xmm1
        xor     eax, eax
1:      vmulps  ymm0, ymm1, ymmword ptr [rcx + rax*4]   // w = e * inv
        vmovups ymmword ptr [rcx + rax*4], ymm0
        add     rax, 8
        cmp     rax, qword ptr [rdx + RC_P]
        jb      1b
        ret

// SM_ONE m, sum, row: 8 keys of one head: e = expf(s - m), stored back, then
// sum += e[0], ..., sum += e[7] (the loads come from the store just made).
.macro SM_ONE m, sum, row
        vmovups ymm0, ymmword ptr [\row + rax*4]
        vsubps  ymm0, ymm0, \m                  // s - m
        EXP8    ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6
        vmovups ymmword ptr [\row + rax*4], ymm0
        vaddss  \sum, \sum, dword ptr [\row + rax*4]
        vaddss  \sum, \sum, dword ptr [\row + rax*4 + 4]
        vaddss  \sum, \sum, dword ptr [\row + rax*4 + 8]
        vaddss  \sum, \sum, dword ptr [\row + rax*4 + 12]
        vaddss  \sum, \sum, dword ptr [\row + rax*4 + 16]
        vaddss  \sum, \sum, dword ptr [\row + rax*4 + 20]
        vaddss  \sum, \sum, dword ptr [\row + rax*4 + 24]
        vaddss  \sum, \sum, dword ptr [\row + rax*4 + 28]
.endm

// softmax_n: k (1..4) score rows XA_SCS4 apart, all with RC's geometry.
//   in: rbx = X, rcx = first row, rdx = RC, r8 = k
//   preserves rdx and callee-saved GPRs; clobbers rax rcx r8-r11, ymm0-6, ymm8-15
// Rows in r8 / r9 / r10 / r11, maxes in ymm8-11, sums in xmm12-15.
        .p2align 4
softmax_n:
        push    rdi
        mov     rdi, r8                         // k
        mov     r8, rcx
        mov     rax, qword ptr [rbx + XA_SCS4]
        lea     r9, [r8 + rax]
        lea     r10, [r9 + rax]
        lea     r11, [r10 + rax]
        call    sm_prep                         // row 0 (rcx)
        vmovaps ymm8, ymm0
        cmp     rdi, 2
        jb      1f
        mov     rcx, r9
        call    sm_prep
        vmovaps ymm9, ymm0
        cmp     rdi, 3
        jb      1f
        mov     rcx, r10
        call    sm_prep
        vmovaps ymm10, ymm0
        cmp     rdi, 4
        jb      1f
        mov     rcx, r11
        call    sm_prep
        vmovaps ymm11, ymm0
1:      vxorps  xmm12, xmm12, xmm12             // sums = 0
        vxorps  xmm13, xmm13, xmm13
        vxorps  xmm14, xmm14, xmm14
        vxorps  xmm15, xmm15, xmm15
        mov     rcx, qword ptr [rdx + RC_P]
        xor     eax, eax
2:      SM_ONE  ymm8, xmm12, r8
        cmp     rdi, 2                          // (predictable branches)
        jb      3f
        SM_ONE  ymm9, xmm13, r9
        cmp     rdi, 3
        jb      3f
        SM_ONE  ymm10, xmm14, r10
        cmp     rdi, 4
        jb      3f
        SM_ONE  ymm11, xmm15, r11
3:      add     rax, 8
        cmp     rax, rcx
        jb      2b
        mov     rcx, r8
        vmovaps xmm0, xmm12
        call    sm_finish
        cmp     rdi, 2
        jb      4f
        mov     rcx, r9
        vmovaps xmm0, xmm13
        call    sm_finish
        cmp     rdi, 3
        jb      4f
        mov     rcx, r10
        vmovaps xmm0, xmm14
        call    sm_finish
        cmp     rdi, 4
        jb      4f
        mov     rcx, r11
        vmovaps xmm0, xmm15
        call    sm_finish
4:      pop     rdi
        ret

// sm_rc: softmax of every head of RC, 4 heads at a time.
//   in: rbx = X, rcx = RC ; preserves callee-saved GPRs
        .p2align 4
sm_rc:
        push    rsi
        push    rdi
        mov     rdx, rcx                        // RC (kept by softmax_n)
        mov     rsi, qword ptr [rdx + RC_G]     // heads left
        mov     rdi, qword ptr [rdx + RC_SC]    // score row of the next head
1:      mov     r8d, 4
        cmp     rsi, r8
        jae     2f
        mov     r8, rsi
2:      sub     rsi, r8
        mov     rcx, rdi
        mov     rax, r8
        imul    rax, qword ptr [rbx + XA_SCS4]
        add     rdi, rax
        call    softmax_n
        test    rsi, rsi
        jnz     1b
        pop     rdi
        pop     rsi
        ret

// zero_o: o[g][s0*16 .. s1*16) = 0 for G heads (clipped to hd).
//   in: rbx = X, rcx = o of head 0, rdx = s0, r8 = s1, r9 = G
//   clobbers rax rcx rdx r8 r9 ymm0
        .p2align 4
zero_o:
        vxorps  ymm0, ymm0, ymm0
        shl     rdx, 6
        shl     r8, 6
        cmp     r8, qword ptr [rbx + XA_HD4]
        jbe     2f
        mov     r8, qword ptr [rbx + XA_HD4]
2:      mov     rax, rdx
3:      vmovups ymmword ptr [rcx + rax], ymm0
        add     rax, 32
        cmp     rax, r8
        jb      3b
        add     rcx, qword ptr [rbx + XA_HD4]
        dec     r9
        jnz     2b
        ret

// ---------------------------------------------------------------------------
// v_abs: o += sum_j w_j * v_j for the keys of [P0, P1) the row attends to
// (clipped to lo..pos), slices [s0, s1), every head of RC's group.
//   in: rbx = X, rcx = RC, rdx = P0, r8 = P1, r9 = s0, r10 = s1
//   preserves callee-saved GPRs
// ---------------------------------------------------------------------------
        .p2align 4
v_abs:
        push    rsi
        push    rdi
        push    rbp
        push    r12
        push    r13
        mov     rsi, rcx
        mov     r12, r9
        mov     r13, r10
        mov     rax, qword ptr [rsi + RC_LO]
        mov     r9, qword ptr [rsi + RC_POS]
        inc     r9
        cmp     rdx, rax
        jae     1f
        mov     rdx, rax                        // a = max(P0, lo)
1:      cmp     r8, r9
        jbe     2f
        mov     r8, r9                          // b = min(P1, pos+1)
2:      cmp     rdx, r8
        jae     .Lva2_ret
        mov     rdi, rdx
        sub     rdi, rax                        // j = a - lo
        mov     rbp, r8
        sub     rbp, rdx                        // keys
        mov     rax, qword ptr [rsi + RC_SLOT0]
        add     rax, rdi
        cmp     rax, qword ptr [rbx + AT_CAP]
        jb      3f
        sub     rax, qword ptr [rbx + AT_CAP]   // slot of a
3:      mov     r8, qword ptr [rbx + AT_CAP]
        sub     r8, rax                         // keys before the ring wraps
        cmp     r8, rbp
        jbe     4f
        mov     r8, rbp
4:      sub     rbp, r8                         // keys after the wrap
        mov     rcx, qword ptr [rsi + RC_OFF]
        add     rcx, rdi
        shl     rcx, 2
        add     rcx, qword ptr [rsi + RC_SC]    // &w[h0][off + j]
        add     rdi, r8                         // j after the wrap
        mov     rdx, rax
        imul    rdx, qword ptr [rbx + XA_KVD4]
        add     rdx, qword ptr [rsi + RC_VB]    // V row of slot a
        mov     r9, qword ptr [rsi + RC_O]
        mov     r10, r12
        mov     r11, r13
        call    v_tile
        test    rbp, rbp
        jz      .Lva2_ret
        mov     rcx, qword ptr [rsi + RC_OFF]
        add     rcx, rdi
        shl     rcx, 2
        add     rcx, qword ptr [rsi + RC_SC]
        mov     rdx, qword ptr [rsi + RC_VB]    // slot 0
        mov     r8, rbp
        mov     r9, qword ptr [rsi + RC_O]
        mov     r10, r12
        mov     r11, r13
        call    v_tile
.Lva2_ret:
        pop     r13
        pop     r12
        pop     rbp
        pop     rdi
        pop     rsi
        ret

// v_tile: n keys contiguous in the ring, slices [s0, s1) (16 floats each) of
// every head of RC. The head_dim columns are walked in chunks of NY ymm
// (8, 4, 2 or 1 ymm = 64..8 floats) and the heads in subgroups of at most 7,
// so that heads x NY accumulators fit in ymm0-13: 1 head x 8, 3 x 4, 7 x 2.
// More accumulators per pass = fewer passes over the V tile and more
// independent FMA chains to hide the 4-cycle FMA latency.
//   in: rbx = X, rsi = RC (for RC_G), rcx = w of head 0 at the first key,
//       rdx = V row of the first key (kv head base), r8 = n, r9 = o of head 0,
//       r10 = s0, r11 = s1
//   preserves callee-saved GPRs
        .p2align 4
v_tile:
        push    rsi
        push    rdi
        push    rbp
        push    r12
        push    r13
        push    r14
        push    r15
        push    qword ptr [rsi + RC_G]          // [rsp+8] = G (rsi = RC on entry)
        push    0                               // [rsp]   = NY of the current column chunk
        mov     rsi, rcx                        // w of head 0
        mov     rdi, rdx                        // V row
        mov     rbp, r8                         // n
        mov     r12, r9                         // o of head 0
        lea     r13, [r10*2]                    // first ymm column
        lea     r15, [r11*2]
        mov     rax, qword ptr [rbx + XA_HD4]
        shr     rax, 5                          // ymm columns per head
        cmp     r15, rax
        jbe     .Lvt_c
        mov     r15, rax                        // end column
.Lvt_c: cmp     r13, r15
        jae     .Lvt_ret
        // NY = widest chunk with min(G,7) * NY <= 14 that fits the columns left
        mov     rax, r15
        sub     rax, r13                        // columns left
        mov     rcx, qword ptr [rsp + 8]        // G
        mov     r11d, 8                         // 1 head: 8 ymm = 64 floats
        cmp     rcx, 1
        jne     2f
        cmp     rax, 8
        jae     5f
2:      mov     r11d, 4                         // <= 3 heads: 4 ymm
        cmp     rcx, 3
        ja      3f
        cmp     rax, 4
        jae     5f
3:      mov     r11d, 2                         // <= 7 heads (per subgroup): 2 ymm
        cmp     rax, 2
        jae     5f
        mov     r11d, 1                         // last 8 floats of hd % 16 == 8
5:      mov     qword ptr [rsp], r11
        xor     r14d, r14d                      // first head of the subgroup
.Lvt_g: mov     r10, qword ptr [rsp + 8]
        sub     r10, r14
        cmp     r10, 7
        jbe     6f
        mov     r10d, 7                         // heads in this subgroup
6:      mov     rcx, r14
        imul    rcx, qword ptr [rbx + XA_SCS4]
        add     rcx, rsi                        // w of the subgroup's first head
        mov     rdx, r13
        shl     rdx, 5
        mov     r9, rdx
        add     rdx, rdi                        // V + column offset
        mov     rax, r14
        imul    rax, qword ptr [rbx + XA_HD4]
        add     r9, rax
        add     r9, r12                         // o + head*hd*4 + column offset
        mov     r8, rbp
        mov     r11, qword ptr [rsp]            // NY
        call    vacc_run
        add     r14, 7
        cmp     r14, qword ptr [rsp + 8]
        jb      .Lvt_g
        add     r13, qword ptr [rsp]            // next column chunk
        jmp     .Lvt_c
.Lvt_ret:
        add     rsp, 16                         // drop NY, G
        pop     r15
        pop     r14
        pop     r13
        pop     r12
        pop     rbp
        pop     rdi
        pop     rsi
        ret

// ---------------------------------------------------------------------------
// vacc_run: the V inner loop. For G heads and NY ymm columns:
//   for j in 0..n-1:  o[g][col] = fma(w[g][j], V[j][col], o[g][col])
// valid shapes: NY = 8 x G = 1, NY = 4 x G <= 3, NY = 2 x G <= 7, NY = 1 x G <= 7.
// The accumulators live in registers for the whole run (loaded from o and
// stored back, so runs can be chained tile after tile).
//   in: rbx = X, rcx = w of head 0 at the first key, rdx = V of the first key
//       (column start), r8 = n, r9 = o of head 0 (column start), r10 = G, r11 = NY
//   preserves callee-saved GPRs; clobbers rax rcx rdx r8-r11, all ymm
// Registers in the loop: accumulator (head g, column y) = ymm(g*NY + y),
// ymm15 = w broadcast; rsi / rdi / r13 = w of heads 0 / 3 / 6, r12 = w stride
// between heads, rdx = V row, rbp = V row stride, rax = prefetch distance,
// r8 = keys left; r10 = o stride between heads (load/store only).
// ---------------------------------------------------------------------------
// (the bodies are generated with .altmacro, where %(expr) turns an assembly-time
//  expression into the number in a register name: ymm%(g*NY+y))
.macro VA_LD r, off                     // accumulator <- o
        vmovups ymm\r, ymmword ptr [r11 + \off]
.endm
.macro VA_ST r, off                     // accumulator -> o
        vmovups ymmword ptr [r11 + \off], ymm\r
.endm
.macro VA_FMA r, off                    // acc = fma(w, V[col], acc)
        vfmadd231ps ymm\r, ymm15, ymmword ptr [rdx + \off]
.endm
.macro VA_W g                           // ymm15 = w of head g at this key
.if \g == 0
        vbroadcastss ymm15, dword ptr [rsi]
.elseif \g == 1
        vbroadcastss ymm15, dword ptr [rsi + r12]
.elseif \g == 2
        vbroadcastss ymm15, dword ptr [rsi + r12*2]
.elseif \g == 3
        vbroadcastss ymm15, dword ptr [rdi]
.elseif \g == 4
        vbroadcastss ymm15, dword ptr [rdi + r12]
.elseif \g == 5
        vbroadcastss ymm15, dword ptr [rdi + r12*2]
.else
        vbroadcastss ymm15, dword ptr [r13]
.endif
.endm
// VA_ALL op, G, NY: op on every accumulator (r11 walks the heads' o)
.macro VA_ALL op, G, NY
        mov     r11, r9
        .set    va_g, 0
        .rept   \G
        .set    va_y, 0
        .rept   \NY
        .altmacro
        \op     %(va_g*\NY + va_y), %(32*va_y)
        .noaltmacro
        .set    va_y, va_y + 1
        .endr
        add     r11, r10
        .set    va_g, va_g + 1
        .endr
.endm
.macro VACC_BODY G, NY
.Lva_\NY\()_\G:
        VA_ALL  VA_LD, \G, \NY
1:      prefetcht0 byte ptr [rdx + rax]         // V row 2^att_vpf_shift keys ahead
.if \NY >= 4
        prefetcht0 byte ptr [rdx + rax + 64]
.endif
.if \NY == 8
        prefetcht0 byte ptr [rdx + rax + 128]
        prefetcht0 byte ptr [rdx + rax + 192]
.endif
        .set    va_g, 0
        .rept   \G
        .altmacro
        VA_W    %va_g
        .noaltmacro
        .set    va_y, 0
        .rept   \NY
        .altmacro
        VA_FMA  %(va_g*\NY + va_y), %(32*va_y)
        .noaltmacro
        .set    va_y, va_y + 1
        .endr
        .set    va_g, va_g + 1
        .endr
        add     rsi, 4                          // next key's weights
.if \G > 3
        add     rdi, 4
.endif
.if \G > 6
        add     r13, 4
.endif
        add     rdx, rbp                        // next V row
        dec     r8
        jnz     1b
        VA_ALL  VA_ST, \G, \NY
        jmp     .Lva_end
.endm

        .p2align 4
vacc_run:
        test    r8, r8
        jz      .Lva_ret0
        push    rsi
        push    rdi
        push    rbp
        push    r12
        push    r13
        mov     rsi, rcx                        // w of head 0
        mov     r12, qword ptr [rbx + XA_SCS4]  // w stride between heads
        lea     rdi, [r12 + r12*2]
        add     rdi, rsi                        // w of head 3
        lea     r13, [rdi + r12*2]
        add     r13, r12                        // w of head 6
        mov     rbp, qword ptr [rbx + XA_KVD4]  // V row stride
        mov     rax, rbp
        mov     rcx, qword ptr [rip + att_vpf_shift]
        shl     rax, cl                         // prefetch distance
        lea     rcx, [r10 - 1]                  // table index: G-1 + base(NY)
        cmp     r11, 2
        jb      1f
        add     rcx, 7                          // NY = 2: 7..13
        cmp     r11, 4
        jb      1f
        add     rcx, 7                          // NY = 4: 14..16
        cmp     r11, 8
        jb      1f
        add     rcx, 3                          // NY = 8: 17
1:      lea     r11, [rip + .Lva_tab]
        movsxd  rcx, dword ptr [r11 + rcx*4]
        add     rcx, r11
        mov     r10, qword ptr [rbx + XA_HD4]   // o stride between heads
        jmp     rcx
        .p2align 2
.Lva_tab:
        .long   .Lva_1_1 - .Lva_tab, .Lva_1_2 - .Lva_tab, .Lva_1_3 - .Lva_tab, .Lva_1_4 - .Lva_tab
        .long   .Lva_1_5 - .Lva_tab, .Lva_1_6 - .Lva_tab, .Lva_1_7 - .Lva_tab
        .long   .Lva_2_1 - .Lva_tab, .Lva_2_2 - .Lva_tab, .Lva_2_3 - .Lva_tab, .Lva_2_4 - .Lva_tab
        .long   .Lva_2_5 - .Lva_tab, .Lva_2_6 - .Lva_tab, .Lva_2_7 - .Lva_tab
        .long   .Lva_4_1 - .Lva_tab, .Lva_4_2 - .Lva_tab, .Lva_4_3 - .Lva_tab
        .long   .Lva_8_1 - .Lva_tab
        .p2align 4
        VACC_BODY 1, 1
        VACC_BODY 2, 1
        VACC_BODY 3, 1
        VACC_BODY 4, 1
        VACC_BODY 5, 1
        VACC_BODY 6, 1
        VACC_BODY 7, 1
        VACC_BODY 1, 2
        VACC_BODY 2, 2
        VACC_BODY 3, 2
        VACC_BODY 4, 2
        VACC_BODY 5, 2
        VACC_BODY 6, 2
        VACC_BODY 7, 2
        VACC_BODY 1, 4
        VACC_BODY 2, 4
        VACC_BODY 3, 4
        VACC_BODY 1, 8
.Lva_end:
        pop     r13
        pop     r12
        pop     rbp
        pop     rdi
        pop     rsi
.Lva_ret0:
        ret
