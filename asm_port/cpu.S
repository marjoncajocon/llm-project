// cpu.S - CPU feature detection for the asm engine (owner: kernels).
//
// Exports
//   i64 cpu_init(void)      detect features, store them in cpu_features, return them
//   i64 cpu_features        (data) the detected bits; dispatch code reads this.
//                           Tests may overwrite it (e.g. clear bit 2 to force
//                           the AVX2-only kernels on a VNNI machine).
//
// Feature bits
//   1 = AVX2 + FMA usable (CPU has them AND the OS saves YMM state: XCR0 bits 1,2)
//   2 = AVX-VNNI (VEX-encoded vpdpbusd, CPUID.(EAX=7,ECX=1):EAX bit 4)
//   4 = WAITPKG (umonitor / umwait / tpause, CPUID.(EAX=7,ECX=0):ECX bit 5):
//       lets the pool's idle workers wait for a new job in a light sleep state
//       (C0.1) that wakes on a write to a watched cache line, instead of
//       spinning with `pause` and stealing issue slots from the SMT sibling.
//
// The bits are only reported as usable when the OS has enabled AVX state
// (CPUID.1:ECX.OSXSAVE and XGETBV(0) & 6 == 6) - otherwise executing a VEX
// instruction would fault even though CPUID advertises it.

.intel_syntax noprefix

#define CPU_AVX2 1
#define CPU_VNNI 2
#define CPU_WAITPKG 4

        .data
        .balign 8
        .globl  cpu_features
cpu_features:
        .quad   0

        .text
// ---------------------------------------------------------------------------
// i64 cpu_init(void)
//   returns rax = feature bits (see above), also stored in cpu_features.
//   clobbers: rax, rcx, rdx, r8, r9 (volatile); rbx is saved (CPUID writes it).
// ---------------------------------------------------------------------------
        .globl  cpu_init
        .def    cpu_init; .scl 2; .type 32; .endef
cpu_init:
        push    rbx
        xor     r8d, r8d                // r8 = result bits

        xor     eax, eax
        cpuid                           // eax = highest standard leaf
        mov     r9d, eax
        cmp     r9d, 7
        jb      .Ldone                  // no leaf 7 -> no AVX2 at all

        mov     eax, 1
        cpuid                           // ecx: bit12 FMA, bit27 OSXSAVE, bit28 AVX
        mov     r10d, ecx
        and     r10d, (1<<12) | (1<<27) | (1<<28)
        cmp     r10d, (1<<12) | (1<<27) | (1<<28)
        jne     .Ldone

        xor     ecx, ecx
        xgetbv                          // edx:eax = XCR0
        and     eax, 6                  // bit1 SSE state, bit2 AVX (YMM upper) state
        cmp     eax, 6
        jne     .Ldone

        mov     eax, 7
        xor     ecx, ecx
        cpuid                           // leaf 7.0: ebx bit5 = AVX2, ecx bit5 = WAITPKG
        test    ecx, 1<<5
        jz      1f
        or      r8d, CPU_WAITPKG        // (an integer feature: no OS state needed)
1:      test    ebx, 1<<5
        jz      .Ldone
        or      r8d, CPU_AVX2

        mov     eax, 7
        mov     ecx, 1
        cpuid                           // leaf 7.1: eax bit4 = AVX-VNNI
        test    eax, 1<<4
        jz      .Ldone
        or      r8d, CPU_VNNI
.Ldone:
        mov     rax, r8
        mov     [rip + cpu_features], rax
        pop     rbx
        ret
