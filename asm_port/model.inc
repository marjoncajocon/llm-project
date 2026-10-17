// model.inc - shared struct offsets for the asm port (see DESIGN.md). Owner: coordinator.
#ifndef MODEL_INC
#define MODEL_INC

// Mat (64 bytes)
#define MAT_KIND    0
#define MAT_K       8
#define MAT_N       16
#define MAT_DATA    24
#define MAT_SCALE   32
#define MAT_PACKED  40
#define MAT_COLSUM  48
#define MAT_AUX     56
#define MAT_SIZE    64
#define MATK_F32    0
#define MATK_F32T   1
#define MATK_Q8     2

// Layer (512 bytes)
#define L_WINDOW    0
#define L_ATTN_NORM 8
#define L_FFN_NORM  16
#define L_BQ        24
#define L_BK        32
#define L_BV        40
#define L_QNORM     48
#define L_KNORM     56
#define L_WQ        64
#define L_WK        128
#define L_WV        192
#define L_WO        256
#define L_WGATE     320
#define L_WUP       384
#define L_WDOWN     448
#define L_SIZE      512

// Model (256 bytes)
#define M_VOCAB     0
#define M_DMODEL    8
#define M_NLAYERS   16
#define M_NHEADS    24
#define M_NKV       32
#define M_HEADDIM   40
#define M_DFF       48
#define M_MAXSEQ    56
#define M_ROPEBASE  64
#define M_EPS       68
#define M_QKNORM    72
#define M_QUANT     80
#define M_TIED      88
#define M_LAYERS    96
#define M_FINALNORM 104
#define M_WTE       112
#define M_LMHEAD    176
#define M_FILEBASE  240
#define M_SPARE     248
#define M_SIZE      256

// KV (64 bytes)
#define KV_NLAYERS  0
#define KV_MAXSEQ   8
#define KV_KVDIM    16
#define KV_LEN      24
#define KV_CAP      32
#define KV_K        40
#define KV_V        48
#define KV_SIZE     64

#endif
