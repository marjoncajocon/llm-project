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
//                               for memory-bound work: the cores of the fastest
//                               efficiency class (the P-cores of a hybrid CPU:
//                               6 on an i5-14400), else all cores.
//   i64  pool_pin, pool_prio    (data, read by the FIRST pool_init) thread
//                               placement / priority, see "Placement" below.
//   u8   pool_cpu_order[]       (data) logical CPU of pool thread i.
//   pool_prof_on, pool_prof[]   (data) optional statistics for tests/dec_test.c.
//
// How it works
//   * A job is published by writing the job fields, then the "claim word"
//     (gen<<32 | chunks_remaining), then bumping pool_gen. Workers wait (umwait,
//     or spin with `pause`) on pool_gen; when it changes they grab chunks with lock cmpxchg on the
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
//   * Idle-but-hot workers wait with umonitor/umwait (WAITPKG, cpu_features
//     bit 4) on pool_gen's cache line instead of a `pause` loop: the core drops
//     into the light C0.1 state and wakes when the caller writes the line, so a
//     waiting thread no longer steals issue slots from the matvec thread on the
//     other half of its P-core. Without WAITPKG the `pause` loop is used.
//   * Placement: pool_init orders the logical CPUs as "one per P-core, then one
//     per E-core, then the SMT siblings" (GetLogicalProcessorInformationEx) and
//     gives pool thread i (the caller is thread 0) CPU pool_cpu_order[i]
//     (pool_pin = 1: ideal processor, a hint the scheduler honours unless that
//     CPU is busy; 2: hard affinity; 0: none). So a decode matvec, which uses
//     threads 0..pool_mem_threads-1, runs on 6 distinct P-cores. The first
//     pool_mem_threads threads also get priority pool_prio (1 = above normal):
//     a dynamically scheduled job still has to wait for every claimed chunk, so
//     a background process that preempts one matvec thread mid-chunk stalls the
//     whole token; above-normal priority stops that (measured: ctx-500 decode
//     under a 2-core background load 26-56 ms -> ~21 ms per token).
//   * Every hot shared variable sits on its own 64-byte cache line.
//   * pool_run must be called from one thread at a time. A call made from
//     inside a running job (fn calling pool_run) is detected (pool_busy) and
//     simply runs inline on that thread.

.intel_syntax noprefix

#define MAXT         64                 // max participating threads
#define SPIN_BUDGET  (1 << 23)          // ~3 ms of TSC ticks before sleeping
#define UMWAIT_TICKS 20000              // max ticks per umwait (~8 us; we re-check after)
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
        // ---- optional statistics (pool_prof_on != 0; read by tests/dec_test.c) ----
        .balign 64
        .globl  pool_prof_on, pool_prof
pool_prof_on:   .quad 0
job_t0:         .quad 0                 // TSC when the current job was published
        .balign 64
pool_prof:                              // [0] parallel runs   [1] ticks inside them
        .space  8*8, 0                  // [2] ticks the caller waited for others
                                        // [3] sum of worker start latencies (ticks)
                                        // [4] number of such samples  [5] SetEvent calls

        .balign 8
        .globl  pool_pin, pool_cpu_order, pool_ncpu_order
pool_pin:       .quad 1                 // place thread i on logical CPU pool_cpu_order[i]:
                                        // 0 = no (OS decides), 1 = ideal processor (soft),
                                        // 2 = affinity mask (hard). Set before pool_init.
pool_prio:      .quad 1                 // priority of the memory-bound threads (see pin_thread)
        .globl  pool_prio
pool_ncpu_order: .quad 0                // entries in pool_cpu_order (0 = topology unknown)

        .bss
        .balign 64
lpi_buf:        .space LPI_BYTES
pool_cpu_order: .space MAXT             // logical CPU numbers (group 0), best first

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
        test    byte ptr [rip + cpu_features], 4        // WAITPKG?
        jz      .Lw_pause
        // umonitor arms a watch on pool_gen's cache line; umwait then sleeps in
        // C0.1 (fast wake-up) until that line is written, or the TSC deadline
        // passes. Re-reading pool_gen AFTER umonitor closes the race with a job
        // posted between the first read and the arm. Unlike a `pause` loop this
        // leaves the core's execution resources to its SMT sibling.
        lea     rcx, [rip + pool_gen]
        umonitor rcx
        mov     eax, [rip + pool_gen]
        cmp     eax, r13d
        jne     .Lw_work
        rdtsc
        shl     rdx, 32
        or      rax, rdx
        mov     rcx, rax
        sub     rcx, r14
        cmp     rcx, SPIN_BUDGET
        jae     .Lw_sleep
        add     rax, UMWAIT_TICKS       // deadline (edx:eax)
        mov     rdx, rax
        shr     rdx, 32
        mov     ecx, 1                  // 1 = C0.1 (lighter state, faster wake-up)
        umwait  ecx
        jmp     .Lw_spin
.Lw_pause:
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
        cmp     qword ptr [rip + pool_prof_on], 0
        je      1f
        rdtsc                                   // start latency of this worker
        shl     rdx, 32
        or      rax, rdx
        sub     rax, [rip + job_t0]
        lock add [rip + pool_prof + 24], rax
        lock inc qword ptr [rip + pool_prof + 32]
1:      mov     ecx, r13d
        call    pool_work
        jmp     .Lw_idle
.Lw_skip:                                       // not in this job: keep spinning but
        cmp     rbx, [rip + pool_active]        // do NOT restart the spin budget, so
        jae     .Lw_sleep                       // a thread left out of a run of jobs
        jmp     .Lw_spin                        // goes to sleep after SPIN_BUDGET

// ---------------------------------------------------------------------------
// pin_thread (internal, Win64): rcx = thread handle, edx = logical CPU number.
//   pool_pin = 1: SetThreadIdealProcessor - a strong hint; the scheduler keeps
//                 the thread there but may still move it when that CPU is busy
//   pool_pin = 2: SetThreadAffinityMask(1 << cpu) - hard: it never moves
// ---------------------------------------------------------------------------
//   r8 = pool index of the thread: the first pool_mem_threads threads (the ones
//   that run the memory-bound decode matvecs) get priority pool_prio
//   (0 = normal, 1 = above normal, 2 = highest) so a busy background process
//   cannot preempt one of them in the middle of its chunk.
        .def    pin_thread; .scl 3; .type 32; .endef
pin_thread:
        push    rbx
        push    rsi
        sub     rsp, 40
        mov     rbx, rcx
        mov     rsi, r8
        cmp     qword ptr [rip + pool_pin], 0
        je      3f
        cmp     qword ptr [rip + pool_pin], 2
        je      1f
        call    SetThreadIdealProcessor
        jmp     3f
1:      mov     eax, edx
        mov     edx, 1
        shlx    rdx, rdx, rax           // affinity mask 1 << cpu
        call    SetThreadAffinityMask
3:      mov     rdx, [rip + pool_prio]
        test    rdx, rdx
        jle     2f
        cmp     rsi, [rip + pool_mem_threads]
        jae     2f
        mov     rcx, rbx
        call    SetThreadPriority
2:      add     rsp, 40
        pop     rsi
        pop     rbx
        ret

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
        // ---- topology (once) -> pool_cpu_order[] and pool_mem_threads ----
        // GetLogicalProcessorInformationEx(RelationProcessorCore = 0, buf, &len)
        // returns one variable-size record per physical core:
        //   +4 Size (u32)   +9 EfficiencyClass (u8: higher = faster core, P > E)
        //   +32 GroupMask[0].Mask (u64: its logical CPUs)   +40 .Group (u16)
        // The order we want for thread i of the pool:
        //   1. one logical CPU of every core of the FASTEST class (the P-cores),
        //   2. one logical CPU of every slower core (E-cores),
        //   3. the remaining SMT siblings.
        // So the first pool_mem_threads threads (the ones memory-bound matvecs use)
        // sit on distinct P-cores, never on an E-core or on two halves of one core.
        mov     [rsp + 56], rbx         // (rbx = n is needed again below)
        mov     dword ptr [rsp + 48], LPI_BYTES
        xor     ecx, ecx                // RelationProcessorCore
        lea     rdx, [rip + lpi_buf]
        lea     r8, [rsp + 48]
        call    GetLogicalProcessorInformationEx
        test    eax, eax
        jz      .Lpi_topo_fail
        mov     r10d, [rsp + 48]
        lea     r11, [rip + lpi_buf]
        add     r10, r11                // r10 = end of the records
        // pass 0: highest efficiency class -> edi
        xor     edi, edi
        mov     rcx, r11
1:      cmp     rcx, r10
        jae     2f
        movzx   eax, byte ptr [rcx + 9]
        cmp     eax, edi
        cmova   edi, eax
        mov     eax, [rcx + 4]
        add     rcx, rax
        jmp     1b
2:      xor     esi, esi                // esi = entries written to pool_cpu_order
        xor     ebx, ebx                // ebx = P-core count (class == max)
        // pass 1 (r9d = 1): fastest class, lowest CPU of each core
        // pass 2 (r9d = 2): slower classes, lowest CPU of each core
        // pass 3 (r9d = 3): all cores, every CPU except the lowest
        mov     r9d, 1
.Lpi_pass:
        mov     rcx, r11
.Lpi_rec:
        cmp     rcx, r10
        jae     .Lpi_pass_next
        cmp     word ptr [rcx + 40], 0  // only processor group 0 (<= 64 CPUs)
        jne     .Lpi_rec_next
        movzx   eax, byte ptr [rcx + 9]
        mov     rdx, [rcx + 32]         // rdx = CPU mask of this core
        cmp     r9d, 3
        je      .Lpi_rest
        cmp     eax, edi                // class == max ?
        sete    al
        cmp     r9d, 1
        sete    ah
        cmp     al, ah                  // pass 1 wants max class, pass 2 the others
        jne     .Lpi_rec_next
        cmp     r9d, 1
        jne     1f
        inc     ebx                     // count P-cores
1:      bsf     rax, rdx                // lowest logical CPU of the core
        jz      .Lpi_rec_next
        cmp     esi, MAXT
        jae     .Lpi_rec_next
        lea     r8, [rip + pool_cpu_order]
        mov     [r8 + rsi], al
        inc     esi
        jmp     .Lpi_rec_next
.Lpi_rest:
        blsr    rdx, rdx                // drop the lowest CPU (already listed)
2:      bsf     rax, rdx
        jz      .Lpi_rec_next
        btr     rdx, rax
        cmp     esi, MAXT
        jae     .Lpi_rec_next
        lea     r8, [rip + pool_cpu_order]
        mov     [r8 + rsi], al
        inc     esi
        jmp     2b
.Lpi_rec_next:
        mov     eax, [rcx + 4]
        add     rcx, rax
        jmp     .Lpi_rec
.Lpi_pass_next:
        inc     r9d
        cmp     r9d, 3
        jbe     .Lpi_pass
        mov     [rip + pool_ncpu_order], rsi
        mov     eax, ebx                // memory-bound work: one thread per fast core
        test    eax, eax
        jnz     1f
.Lpi_topo_fail:
        mov     eax, 4                  // query failed: a safe guess
1:      mov     [rip + pool_mem_threads], rax
        // pin the calling thread (participant 0) as well
        cmp     qword ptr [rip + pool_ncpu_order], 0
        je      .Lpi_topo_restore
        call    GetCurrentThread
        mov     rcx, rax
        movzx   edx, byte ptr [rip + pool_cpu_order]
        xor     r8d, r8d                // index 0 = the caller
        call    pin_thread
.Lpi_topo_restore:
        mov     rbx, [rsp + 56]
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
        // pin worker i to logical CPU pool_cpu_order[i] (and set its priority)
        cmp     rdi, [rip + pool_ncpu_order]
        jae     1f
        mov     rcx, rax
        lea     rdx, [rip + pool_cpu_order]
        movzx   edx, byte ptr [rdx + rdi]
        mov     r8, rdi
        call    pin_thread
1:
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
        cmp     qword ptr [rip + pool_prof_on], 0
        je      1f
        rdtsc
        shl     rdx, 32
        or      rax, rdx
        mov     [rip + job_t0], rax
        inc     qword ptr [rip + pool_prof]
1:      mov     eax, ebx
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
        inc     qword ptr [rip + pool_prof + 40]
.Lpr_wnext:
        inc     rsi
        add     rdi, SLOT_SIZE
        jmp     .Lpr_wake
.Lpr_work:
        mov     ecx, ebx
        call    pool_work               // the caller works too
        cmp     qword ptr [rip + pool_prof_on], 0
        jne     .Lpr_wait_prof
.Lpr_wait:
        cmp     [rip + pool_done], r12
        je      .Lpr_end
        pause
        jmp     .Lpr_wait
.Lpr_wait_prof:                         // same wait, but timed
        rdtsc
        shl     rdx, 32
        or      rax, rdx
        mov     rsi, rax
1:      cmp     [rip + pool_done], r12
        je      2f
        pause
        jmp     1b
2:      rdtsc
        shl     rdx, 32
        or      rax, rdx
        mov     rcx, rax
        sub     rax, rsi
        add     [rip + pool_prof + 16], rax
        sub     rcx, [rip + job_t0]
        add     [rip + pool_prof + 8], rcx
.Lpr_end:
        mov     qword ptr [rip + pool_busy], 0
        add     rsp, 40
        pop     r12
        pop     rdi
        pop     rsi
        pop     rbx
        ret
