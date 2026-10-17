// pool.S - persistent low-latency thread pool (owner: kernels).
//
// Exports (Win64 ABI)
//   i64  pool_init(i64 n)       create/resize the pool to n participating threads
//                               (the calling thread counts as one). n <= 0 = auto.
//                               May be called again later (only while no job runs)
//                               to change the count; returns the new count.
//   void pool_run(fn, ctx, i64 n, i64 grain)
//                               split [0,n) into ceil(n/grain) chunks and call
//                               fn(ctx, i0, i1) (Win64) on each, using all
//                               participating threads; the caller works too and
//                               pool_run returns when every chunk has finished.
//   i64  pool_threads(void)     number of participating threads (1 before init)
//   void pool_run_n(fn, ctx, n, grain, i64 nthreads)
//                               pool_run with at most nthreads participants
//                               (<= 0: all). Used by matvec for memory-bound
//                               work, where fewer threads stream DRAM better.
//   i64  pool_mem_threads       (data, set by pool_init) suggested thread count
//                               for memory-bound work: cores with SMT (the
//                               P-cores of a hybrid CPU: 6 on an i5-14400),
//                               else all cores.
//
// How it works
//   * A job is published by writing the job fields, then the "claim word"
//     (gen<<32 | chunks_remaining), then bumping pool_gen. Workers spin (with `pause`)
//     on pool_gen; when it changes they grab chunks with lock cmpxchg on the
//     claim word. Embedding the generation in the claim word makes a straggler
//     from an older job unable to steal a chunk of a newer job.
//   * Chunk scheduling is dynamic (first come first served), which balances the
//     hybrid P/E cores of a Raptor Lake automatically.
//   * Completion = a per-job counter of finished chunks; the caller spins until
//     it equals nchunks. A worker reads fn/ctx/n/grain only AFTER it has claimed
//     a chunk, and the caller cannot start the next job before that chunk is
//     done, so the fields it reads are always those of its own job.
//   * After SPIN_BUDGET TSC ticks without work a worker goes to sleep on its own
//     auto-reset event (Dekker handshake with its `sleeping` flag: the worker
//     sets the flag then re-checks pool_gen; the caller bumps pool_gen with a
//     locked xchg then checks the flag - both sides are full fences, so a
//     wake-up can never be lost). Idle threads therefore stop burning CPU a few
//     ms after the last job, while back-to-back matvecs (a decode step issues
//     ~170) see only the spin-path latency (~100s of ns).
//   * Every hot shared variable sits on its own 64-byte cache line.
//   * pool_run must be called from one thread at a time. A call made from
//     inside a running job (fn calling pool_run) is detected (pool_busy) and
//     simply runs inline on that thread.

.intel_syntax noprefix

#define MAXT         64                 // max participating threads
#define SPIN_BUDGET  (1 << 23)          // ~3 ms of TSC ticks before sleeping
#define SLOT_SLEEP   0                  // per-worker slot (64 B, own line)
#define SLOT_EVENT   8
#define SLOT_THREAD  16
#define SLOT_SIZE    64
#define LPI_BYTES    8192               // GetLogicalProcessorInformation buffer

        .data
        .balign 64
        // exported for the test watchdog (debug dumps), not an API:
        .globl pool_active, pool_gen, pool_claim, pool_done, job_nchunks, pool_slots
        .globl pool_mem_threads
pool_active:    .quad 1                 // participating threads incl. caller
pool_created:   .quad 0                 // worker threads created so far
pool_busy:      .quad 0                 // 1 while a parallel job is in flight
pool_mem_threads: .quad 0               // suggested threads for memory-bound work
                                        // (= SMT cores, i.e. P-cores on a hybrid
                                        //  CPU; else all cores); set by pool_init
        .balign 64
// ---- job line: written by the caller once per job, read by all workers ----
pool_gen:       .quad 0                 // generation (low 32 bits used)
job_fn:         .quad 0
job_ctx:        .quad 0
job_n:          .quad 0
job_grain:      .quad 0
job_nchunks:    .quad 0
job_limit:      .quad 0                 // workers with index >= limit sit this job out
        .balign 64
pool_claim:     .quad 0                 // gen<<32 | chunks remaining (contended)
        .balign 64
pool_done:      .quad 0                 // chunks finished in the current job
        .balign 64
pool_slots:     .space MAXT*SLOT_SIZE, 0

        .bss
        .balign 64
lpi_buf:        .space LPI_BYTES

        .text

// ---------------------------------------------------------------------------
// pool_work (internal): claim and run chunks of generation ecx until none left.
//   in : ecx = generation the caller/worker observed
//   clobbers: all volatile regs (it calls fn); saves rbx, rsi
// ---------------------------------------------------------------------------
        .def    pool_work; .scl 3; .type 32; .endef
pool_work:
        push    rbx
        push    rsi
        sub     rsp, 40                 // shadow space + keep rsp 16-aligned
        mov     ebx, ecx                // ebx = our generation
        mov     rax, [rip + pool_claim]
.Lpw_claim:
        // claim word = gen<<32 | chunks_remaining. Both halves are validated by
        // the same lock cmpxchg, so a straggler can never take a chunk of a
        // finished job (an earlier version compared the index with job_nchunks
        // read separately - that value could already belong to the NEXT job,
        // letting a phantom chunk run and over-count pool_done -> hang).
        mov     rdx, rax
        shr     rdx, 32
        cmp     edx, ebx                // claim word belongs to another job?
        jne     .Lpw_ret
        test    eax, eax                // chunks remaining?
        jz      .Lpw_ret
        lea     rdx, [rax - 1]
        lock cmpxchg [rip + pool_claim], rdx   // rax reloaded on failure
        jne     .Lpw_claim
        // we own chunk (nchunks - remaining) of this job; the job cannot finish
        // before we do, so its fields are stable now
        mov     esi, eax
        neg     rsi
        add     rsi, [rip + job_nchunks] // idx
        mov     rax, [rip + job_grain]
        imul    rsi, rax                // i0 = idx * grain
        lea     r8, [rsi + rax]         // i1 = i0 + grain
        mov     rax, [rip + job_n]
        cmp     r8, rax
        cmova   r8, rax                 // i1 = min(i1, n)
        mov     rdx, rsi
        mov     rcx, [rip + job_ctx]
        call    [rip + job_fn]          // fn(ctx, i0, i1)
        lock inc qword ptr [rip + pool_done]
        mov     rax, [rip + pool_claim]
        jmp     .Lpw_claim
.Lpw_ret:
        add     rsp, 40
        pop     rsi
        pop     rbx
        ret

// ---------------------------------------------------------------------------
// worker_proc (internal thread entry): DWORD WINAPI worker_proc(void* index)
//   rbx = worker index (1..MAXT-1), r12 = its slot, r13 = last seen generation
// ---------------------------------------------------------------------------
        .def    worker_proc; .scl 3; .type 32; .endef
worker_proc:
        push    rbx
        push    r12
        push    r13
        push    r14
        push    r15
        sub     rsp, 32                 // 5 pushes + ret addr = 48 -> aligned after sub 32
        mov     rbx, rcx
        lea     r12, [rip + pool_slots]
        mov     rax, rcx
        shl     rax, 6
        add     r12, rax                // r12 = &pool_slots[index]
        mov     r13d, [rip + pool_gen]  // start from the current generation
.Lw_idle:
        cmp     rbx, [rip + pool_active]
        jae     .Lw_sleep               // not participating: sleep right away
        rdtsc
        shl     rdx, 32
        or      rax, rdx
        mov     r14, rax                // r14 = spin start time
.Lw_spin:
        mov     eax, [rip + pool_gen]
        cmp     eax, r13d
        jne     .Lw_work
        pause
        rdtsc
        shl     rdx, 32
        or      rax, rdx
        sub     rax, r14
        cmp     rax, SPIN_BUDGET
        jb      .Lw_spin
.Lw_sleep:
        mov     eax, 1
        xchg    [r12 + SLOT_SLEEP], rax         // announce sleep (full fence)
        mov     eax, [rip + pool_gen]           // re-check: job posted meanwhile?
        cmp     eax, r13d
        jne     .Lw_wake_now
        mov     rcx, [r12 + SLOT_EVENT]
        mov     edx, -1                         // INFINITE
        call    WaitForSingleObject
        mov     qword ptr [r12 + SLOT_SLEEP], 0
        mov     eax, [rip + pool_gen]
        cmp     eax, r13d
        jne     .Lw_work
        jmp     .Lw_idle                        // spurious (stale signal)
.Lw_wake_now:
        mov     qword ptr [r12 + SLOT_SLEEP], 0
.Lw_work:
        mov     r13d, eax                       // remember this generation
        cmp     rbx, [rip + job_limit]
        jae     .Lw_skip
        mov     ecx, eax
        call    pool_work
        jmp     .Lw_idle
.Lw_skip:                                       // not in this job: keep spinning but
        cmp     rbx, [rip + pool_active]        // do NOT restart the spin budget, so
        jae     .Lw_sleep                       // a thread left out of a run of jobs
        jmp     .Lw_spin                        // goes to sleep after SPIN_BUDGET

// ---------------------------------------------------------------------------
// i64 pool_init(i64 n)
//   n <= 0: auto = number of logical processors (GetActiveProcessorCount).
//   Clamped to [1, MAXT]. Creates missing worker threads, sets pool_active.
//   returns rax = participating thread count. Saves rbx, rsi, rdi.
// ---------------------------------------------------------------------------
        .globl  pool_init
        .def    pool_init; .scl 2; .type 32; .endef
pool_init:
        push    rbx
        push    rsi
        push    rdi
        sub     rsp, 64                 // 3 pushes + ret = 32 -> +64 keeps 16-align; 48 B for 6 args
        mov     rbx, rcx
        cmp     qword ptr [rip + pool_mem_threads], 0
        jne     .Lpi_topo_done
        // topology: count cores and SMT cores (GetLogicalProcessorInformation,
        // 32-byte records; Relationship at +8 (0 = ProcessorCore), Flags at +16
        // (1 = the core runs 2 hardware threads)).
        mov     dword ptr [rsp + 48], LPI_BYTES
        lea     rcx, [rip + lpi_buf]
        lea     rdx, [rsp + 48]
        call    GetLogicalProcessorInformation
        xor     esi, esi                // cores
        xor     edi, edi                // SMT cores
        test    eax, eax
        jz      .Lpi_topo_set
        mov     eax, [rsp + 48]
        lea     rcx, [rip + lpi_buf]
        lea     rdx, [rcx + rax]
.Lpi_topo_loop:
        cmp     rcx, rdx
        jae     .Lpi_topo_set
        cmp     dword ptr [rcx + 8], 0
        jne     .Lpi_topo_next
        inc     rsi
        cmp     byte ptr [rcx + 16], 1
        jne     .Lpi_topo_next
        inc     rdi
.Lpi_topo_next:
        add     rcx, 32
        jmp     .Lpi_topo_loop
.Lpi_topo_set:
        mov     rax, rdi                // prefer SMT (= P-) cores
        test    rax, rax
        cmovz   rax, rsi                // no SMT: all cores
        test    rax, rax
        jnz     1f
        mov     eax, 4                  // query failed: a safe guess
1:      mov     [rip + pool_mem_threads], rax
.Lpi_topo_done:
        test    rbx, rbx
        jg      .Lpi_have
        mov     ecx, 0xFFFF             // ALL_PROCESSOR_GROUPS
        call    GetActiveProcessorCount
        mov     ebx, eax
.Lpi_have:
        cmp     rbx, 1
        jge     1f
        mov     ebx, 1
1:      cmp     rbx, MAXT
        jle     1f
        mov     ebx, MAXT
1:
        // create workers pool_created+1 .. rbx-1
.Lpi_create:
        mov     rsi, [rip + pool_created]
        lea     rax, [rsi + 1]
        cmp     rax, rbx
        jge     .Lpi_done
        mov     rdi, rax                // rdi = new worker index
        // event = CreateEventW(NULL, bManualReset=FALSE, bInitialState=FALSE, NULL)
        xor     ecx, ecx
        xor     edx, edx
        xor     r8d, r8d
        xor     r9d, r9d
        call    CreateEventW
        lea     rcx, [rip + pool_slots]
        mov     rdx, rdi
        shl     rdx, 6
        mov     [rcx + rdx + SLOT_EVENT], rax
        mov     qword ptr [rcx + rdx + SLOT_SLEEP], 0
        // CreateThread(NULL, 0, worker_proc, index, 0, NULL)
        xor     ecx, ecx
        xor     edx, edx
        lea     r8, [rip + worker_proc]
        mov     r9, rdi
        mov     qword ptr [rsp + 32], 0
        mov     qword ptr [rsp + 40], 0
        call    CreateThread
        lea     rcx, [rip + pool_slots]
        mov     rdx, rdi
        shl     rdx, 6
        mov     [rcx + rdx + SLOT_THREAD], rax
        mov     [rip + pool_created], rdi
        jmp     .Lpi_create
.Lpi_done:
        mov     [rip + pool_active], rbx
        mov     rax, rbx
        add     rsp, 64
        pop     rdi
        pop     rsi
        pop     rbx
        ret

// ---------------------------------------------------------------------------
// i64 pool_threads(void)  -> rax = participating threads (incl. caller)
// ---------------------------------------------------------------------------
        .globl  pool_threads
        .def    pool_threads; .scl 2; .type 32; .endef
pool_threads:
        mov     rax, [rip + pool_active]
        ret

// ---------------------------------------------------------------------------
// void pool_run(void (*fn)(void* ctx, i64 i0, i64 i1), void* ctx, i64 n, i64 grain)
//   rcx = fn, rdx = ctx, r8 = n, r9 = grain (<=0 treated as 1)
//   Runs inline (single call fn(ctx,0,n)) when the pool has 1 thread or the
//   whole range is one chunk. Saves rbx, rsi, rdi, r12.
// void pool_run_n(fn, ctx, n, grain, i64 nthreads)   ([rsp+40] = nthreads)
//   same, but at most nthreads threads (incl. the caller) take part;
//   nthreads <= 0 or > pool_threads() means all.
// ---------------------------------------------------------------------------
        .globl  pool_run_n
        .def    pool_run_n; .scl 2; .type 32; .endef
pool_run_n:
        mov     r10, [rsp + 40]
        cmp     r10, [rip + pool_active]
        jg      1f
        test    r10, r10
        jg      2f
1:      mov     r10, [rip + pool_active]
2:      jmp     .Lpr_common
        .globl  pool_run
        .def    pool_run; .scl 2; .type 32; .endef
pool_run:
        mov     r10, [rip + pool_active]        // r10 = threads for this job
.Lpr_common:
        test    r8, r8
        jle     .Lpr_ret0
        test    r9, r9
        jg      1f
        mov     r9d, 1
1:      cmp     r10, 1
        jle     .Lpr_inline
        cmp     qword ptr [rip + pool_busy], 0  // nested call from inside a job
        jne     .Lpr_inline                     //   (e.g. fn calls matvec): inline
        cmp     r8, r9
        jg      .Lpr_parallel
.Lpr_inline:                            // tail call fn(ctx, 0, n)
        mov     rax, rcx
        mov     rcx, rdx
        xor     edx, edx
        jmp     rax
.Lpr_ret0:
        ret
.Lpr_parallel:
        push    rbx
        push    rsi
        push    rdi
        push    r12
        sub     rsp, 40                 // 4 pushes + ret = 40 -> +40 = 80, aligned
        mov     qword ptr [rip + pool_busy], 1
        // publish the job fields
        mov     [rip + job_fn], rcx
        mov     [rip + job_ctx], rdx
        mov     [rip + job_n], r8
        mov     [rip + job_grain], r9
        mov     [rip + job_limit], r10
        lea     rax, [r8 + r9 - 1]
        xor     edx, edx
        div     r9                      // rax = nchunks
        mov     [rip + job_nchunks], rax
        mov     r12, rax
        mov     qword ptr [rip + pool_done], 0
        mov     ebx, [rip + pool_gen]
        inc     ebx                     // ebx = new generation (32-bit wrap)
        mov     rax, rbx
        shl     rax, 32
        or      rax, r12                // gen<<32 | nchunks remaining
        mov     [rip + pool_claim], rax // claim word before the generation
        mov     eax, ebx
        xchg    [rip + pool_gen], rax   // publish (locked = full fence)
        // wake participating workers that went to sleep
        mov     rsi, 1
        lea     rdi, [rip + pool_slots + SLOT_SIZE]
.Lpr_wake:
        cmp     rsi, [rip + job_limit]
        jae     .Lpr_work
        cmp     qword ptr [rdi + SLOT_SLEEP], 0
        je      .Lpr_wnext
        xor     eax, eax
        xchg    [rdi + SLOT_SLEEP], rax
        test    rax, rax
        jz      .Lpr_wnext
        mov     rcx, [rdi + SLOT_EVENT]
        call    SetEvent
.Lpr_wnext:
        inc     rsi
        add     rdi, SLOT_SIZE
        jmp     .Lpr_wake
.Lpr_work:
        mov     ecx, ebx
        call    pool_work               // the caller works too
.Lpr_wait:
        cmp     [rip + pool_done], r12
        je      .Lpr_end
        pause
        jmp     .Lpr_wait
.Lpr_end:
        mov     qword ptr [rip + pool_busy], 0
        add     rsp, 40
        pop     r12
        pop     rdi
        pop     rsi
        pop     rbx
        ret
