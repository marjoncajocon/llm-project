// kern_stream.S - test helper (not part of the product): a plain streaming
// read used as the memory-bandwidth yardstick for the matvec benchmarks.
//
//   u64 stream_read(const void* p, i64 bytes)   p 64-byte aligned, bytes % 256 == 0
//   returns the XOR of all 32-byte lanes folded to 64 bits (keeps the loads live).
//   clobbers rax, rcx, rdx, ymm0-ymm3 (volatile only).
.intel_syntax noprefix
        .text
        .globl  stream_read
stream_read:
        vpxor   ymm0, ymm0, ymm0
        vpxor   ymm1, ymm1, ymm1
        vpxor   ymm2, ymm2, ymm2
        vpxor   ymm3, ymm3, ymm3
        add     rdx, rcx                    // end
        .p2align 5
1:      cmp     rcx, rdx
        jae     2f
        vpxor   ymm0, ymm0, [rcx]
        vpxor   ymm1, ymm1, [rcx + 32]
        vpxor   ymm2, ymm2, [rcx + 64]
        vpxor   ymm3, ymm3, [rcx + 96]
        vpxor   ymm0, ymm0, [rcx + 128]
        vpxor   ymm1, ymm1, [rcx + 160]
        vpxor   ymm2, ymm2, [rcx + 192]
        vpxor   ymm3, ymm3, [rcx + 224]
        add     rcx, 256
        jmp     1b
2:      vpxor   ymm0, ymm0, ymm1
        vpxor   ymm2, ymm2, ymm3
        vpxor   ymm0, ymm0, ymm2
        vextracti128 xmm1, ymm0, 1
        vpxor   xmm0, xmm0, xmm1
        vpsrldq xmm1, xmm0, 8
        vpxor   xmm0, xmm0, xmm1
        vmovq   rax, xmm0
        vzeroupper
        ret
