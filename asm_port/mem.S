// mem.S - memory allocation for the asm port (owner: loader agent)
//
//   void* mem_alloc(i64 bytes)            64-byte aligned, ZEROED memory (0 on failure)
//   void  mem_free(void* p)               p from mem_alloc (0 is ignored)
//   void  mem_decommit(void* p, i64 n)    give the whole 4 KB pages inside [p, p+n) back
//                                         to the OS (big blocks only; the range must not
//                                         be touched again). Used by the loader to drop
//                                         raw int8 weights once the kernels packed them.
//   void  mem_copy(void* dst, const void* src, i64 n)   plain forward copy (rep movsb)
//
// How it works
// ------------
// Small blocks (< 1 MB) come from the process heap (HeapAlloc with HEAP_ZERO_MEMORY);
// big blocks come straight from VirtualAlloc, which always hands out zero-filled pages
// (and only charges physical memory when a page is first touched -- that is why a
// 800 MB KV cache costs nothing until it is used).
//
// Every block carries a 16-byte header just BELOW the pointer we return:
//
//      base ... [p-16] = base pointer to hand back to HeapFree / VirtualFree
//               [p-8]  = kind: 1 = heap, 2 = VirtualAlloc, 3 = VirtualAlloc large pages
//               [p]    = 64-byte aligned user memory
//
// Large pages (2 MB TLB entries) need the SeLockMemoryPrivilege, which a normal user
// does not hold (and enabling it needs advapi32, which the port does not link). We try
// MEM_LARGE_PAGES exactly once for the first big block; if Windows refuses we remember
// that and never try again, so the cost is one failed syscall.
//
// Win64 ABI reminder: args in rcx, rdx, r8, r9; caller reserves 32 bytes of "shadow
// space" above the return address; rsp must be 16-byte aligned at every `call`.
// At function entry rsp is 8 mod 16 (the call pushed the return address), so
// "N pushes + sub rsp, S" must make 8 + 8*N + S a multiple of 16.

    .intel_syntax noprefix
#include "model.inc"

#define MEM_COMMIT        0x1000
#define MEM_RESERVE       0x2000
#define MEM_DECOMMIT      0x4000
#define MEM_RELEASE       0x8000
#define MEM_LARGE_PAGES   0x20000000
#define PAGE_READWRITE    0x04
#define HEAP_ZERO_MEMORY  0x08
#define BIG_THRESHOLD     (1 << 20)     // >= 1 MB goes to VirtualAlloc
#define HDR               64            // bytes in front of a VirtualAlloc block

    .bss
    .balign 8
mem_heap:      .quad 0                  // GetProcessHeap() result, fetched lazily
mem_lp_state:  .quad 0                  // large pages: 0 untried, 1 work, 2 refused

    .text

// ---------------------------------------------------------------------------
// void* mem_alloc(i64 bytes)
// ---------------------------------------------------------------------------
    .globl mem_alloc
mem_alloc:
    push rbx
    push rsi
    push rdi
    sub rsp, 32                         // 8 + 3*8 + 32 = 64: aligned for calls
    mov rbx, rcx                        // rbx = requested size
    test rbx, rbx
    jns 0f
    xor ebx, ebx                        // negative size: treat as 0
0:  cmp rbx, BIG_THRESHOLD
    jae .Lma_big

    // ---- small block: process heap ----
    mov rcx, qword ptr [rip + mem_heap]
    test rcx, rcx
    jnz 1f
    call qword ptr [rip + __imp_GetProcessHeap]
    mov qword ptr [rip + mem_heap], rax
    mov rcx, rax
1:  mov edx, HEAP_ZERO_MEMORY           // HeapAlloc(heap, HEAP_ZERO_MEMORY, size)
    lea r8, [rbx + 80]                  // + 16 header + up to 63 alignment slack
    call qword ptr [rip + __imp_HeapAlloc]
    test rax, rax
    jz .Lma_fail
    lea rdi, [rax + 16 + 63]            // leave room for the header, then
    and rdi, -64                        // round up to a 64-byte boundary
    mov qword ptr [rdi - 16], rax       // header: base
    mov qword ptr [rdi - 8], 1          // header: kind = heap
    mov rax, rdi
    jmp .Lma_done

    // ---- big block: VirtualAlloc ----
.Lma_big:
    lea rsi, [rbx + HDR]                // total bytes incl. the header page prefix
    cmp qword ptr [rip + mem_lp_state], 2
    je .Lma_plain
    cmp rsi, (4 << 20)                  // only worth trying for >= 4 MB blocks
    jb .Lma_plain
    call qword ptr [rip + __imp_GetLargePageMinimum]
    test rax, rax
    jz .Lma_lp_refused                  // no large-page support at all
    lea rdx, [rsi + rax - 1]            // round size up to a large-page multiple
    neg rax
    and rdx, rax
    xor ecx, ecx                        // VirtualAlloc(NULL, size, flags, PAGE_READWRITE)
    mov r8d, MEM_COMMIT | MEM_RESERVE | MEM_LARGE_PAGES
    mov r9d, PAGE_READWRITE
    call qword ptr [rip + __imp_VirtualAlloc]
    test rax, rax
    jz .Lma_lp_refused
    mov qword ptr [rip + mem_lp_state], 1
    lea rdi, [rax + HDR]
    mov qword ptr [rdi - 16], rax
    mov qword ptr [rdi - 8], 3          // kind = large-page VirtualAlloc
    mov rax, rdi
    jmp .Lma_done
.Lma_lp_refused:
    mov qword ptr [rip + mem_lp_state], 2   // never try again
.Lma_plain:
    xor ecx, ecx
    mov rdx, rsi
    mov r8d, MEM_COMMIT | MEM_RESERVE
    mov r9d, PAGE_READWRITE
    call qword ptr [rip + __imp_VirtualAlloc]
    test rax, rax
    jz .Lma_fail
    lea rdi, [rax + HDR]                // VirtualAlloc is 64 KB aligned, so +64 is 64-aligned
    mov qword ptr [rdi - 16], rax
    mov qword ptr [rdi - 8], 2          // kind = VirtualAlloc
    mov rax, rdi
    jmp .Lma_done
.Lma_fail:
    xor eax, eax
.Lma_done:
    add rsp, 32
    pop rdi
    pop rsi
    pop rbx
    ret

// ---------------------------------------------------------------------------
// void mem_free(void* p)
// ---------------------------------------------------------------------------
    .globl mem_free
mem_free:
    sub rsp, 40                         // 8 + 40 = 48: aligned
    test rcx, rcx
    jz .Lmf_ret
    mov rax, qword ptr [rcx - 16]       // base
    mov rdx, qword ptr [rcx - 8]        // kind
    cmp rdx, 1
    je .Lmf_heap
    mov rcx, rax                        // VirtualFree(base, 0, MEM_RELEASE)
    xor edx, edx
    mov r8d, MEM_RELEASE
    call qword ptr [rip + __imp_VirtualFree]
    jmp .Lmf_ret
.Lmf_heap:
    mov r8, rax                         // HeapFree(heap, 0, base)
    mov rcx, qword ptr [rip + mem_heap]
    xor edx, edx
    call qword ptr [rip + __imp_HeapFree]
.Lmf_ret:
    add rsp, 40
    ret

// ---------------------------------------------------------------------------
// void mem_decommit(void* p, i64 n): VirtualFree(MEM_DECOMMIT) on the full pages
// strictly inside [p, p+n). Partial pages at either end are kept (they may hold
// neighbouring data). Failure (e.g. a heap or large-page block) is ignored.
// ---------------------------------------------------------------------------
    .globl mem_decommit
mem_decommit:
    sub rsp, 40
    lea rax, [rcx + rdx]                // end = p + n
    and rax, -4096                      // round end DOWN to a page
    add rcx, 4095                       // round start UP to a page
    and rcx, -4096
    mov rdx, rax
    sub rdx, rcx                        // size = end - start
    jle .Lmd_ret                        // no whole page inside
    mov r8d, MEM_DECOMMIT               // VirtualFree(start, size, MEM_DECOMMIT)
    call qword ptr [rip + __imp_VirtualFree]
.Lmd_ret:
    add rsp, 40
    ret

// ---------------------------------------------------------------------------
// void mem_copy(void* dst, const void* src, i64 n)
// `rep movsb` copies rcx bytes from [rsi] to [rdi]; on modern CPUs ("ERMSB") it is
// as fast as a hand-written AVX loop for big blocks. rsi/rdi are callee-saved in
// Win64, so we save them around it.
// ---------------------------------------------------------------------------
    .globl mem_copy
mem_copy:
    push rsi
    push rdi
    mov rdi, rcx
    mov rsi, rdx
    mov rcx, r8
    test rcx, rcx
    jle 1f
    rep movsb
1:  pop rdi
    pop rsi
    ret
