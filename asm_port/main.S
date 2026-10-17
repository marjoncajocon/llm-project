// main.S - the command line program: a drop-in for `transformer hfgen`
// (owner: loader agent)
//
//   llm.exe <model.bin> <tok.hf> [n_new] [temp] [seed] [--raw] [--bench]
//           [--threads N] [--q8]
//
// Port of ../main.c run_hfgen + ../gen.c hfgen_file / dec_generate / hf_stop /
// hf_emit_delta for ONE sample, no constraint, no stop-tags:
//
//   1. read the WHOLE prompt from stdin (CRT text-mode rules: "\r\n" -> "\n",
//      Ctrl-Z ends it), cut at the first NUL (C passes it as a C string), tokenize
//   2. load the model (LLMA fp32 or LLMQ int8; GPTC is not supported here)
//   3. fit the prompt into the context (hfgen_fit_prompt): if prompt + reply do not
//      fit, drop the OLDEST tokens, keeping max_seq - clamp(n_new, 1, max_seq/2)
//   4. prefill, then up to n_new times: sample (top_k 40, top_p 0.95, temp, seed),
//      stop on an end-of-generation token, append the token's bytes to the decoded
//      text, cut the text at an end marker or a hallucinated next-turn label
//      (hf_stop), print the NEW bytes as {"d":"..."} and decode the token
//   5. {"done":true}
//
// Output is byte-identical to the C engine's, including its "\r\n" line ends
// (the C runtime writes stdout in text mode). Errors: a message on stderr
// ("hfgen: error: ..."), then {"error":"..."} and {"done":true} on stdout.
// --raw writes the generated text itself (no JSON, no done line); --bench adds
// timings on stderr; --threads sizes the thread pool; --q8 quantizes an fp32
// checkpoint to int8 at load.
//
// Functions used from the other modules (DESIGN.md):
//   cpu_init, pool_init, pool_threads              (kernels: cpu.S, pool.S)
//   tok_load, tok_encode, tok_piece, tok_is_eog,
//   tok_vocab, tok_max_piece                      (tokenizer: tok.S)
//   decode_step, prefill, rng_seed, sample_token  (decode: decode.S, sample.S)
//   model_load, model_peek_magic, kv_create_len,
//   mem_*, io_*, msg_*, json_escape, parse_*      (this agent)

    .intel_syntax noprefix
#include "model.inc"

#define LLMA_MAGIC   0x414D4C4C
#define LLMQ_MAGIC   0x514D4C4C
#define GPTC_MAGIC   0x43545047
#define DEC_CAP      (1 << 20)          // gen.c: char dec[1 << 20] (holds 2^20 - 1 bytes)
#define TOP_K        40
#define MAX_POS      8

    .section .rdata, "dr"
    .balign 4
f_top_p:     .float 0.95
f_temp_def:  .float 0.8
    .balign 8
d_1e9:       .double 1.0e9

s_raw:          .asciz "--raw"
s_bench:        .asciz "--bench"
s_q8:           .asciz "--q8"
s_threads:      .asciz "--threads"
s_u_stop:       .asciz "--stop-tags"
s_u_samples:    .asciz "--samples"
s_u_con:        .asciz "--constrain"
s_u_conb:       .asciz "--constrain-body"
s_u_conc:       .asciz "--constrain-cmd"
s_u_cont:       .asciz "--constrain-think"
    .balign 8
unsup_tab:      .quad s_u_stop, s_u_samples, s_u_con, s_u_conb, s_u_conc, s_u_cont, 0

s_usage:
    .ascii "usage: llm <model.bin> <tok.hf> [n_new] [temp] [seed] [--raw] [--bench] [--threads N] [--q8]\r\n"
    .ascii "  A pure x86-64 assembly drop-in for `transformer hfgen`: reads the prompt from\r\n"
    .ascii "  stdin and streams the reply as {\"d\":\"...\"} JSON lines, then {\"done\":true}.\r\n"
    .ascii "  n_new defaults to 128, temp to 0.8 (<= 0: greedy), seed to 1234.\r\n"
    .ascii "  --raw        print the generated text itself instead of JSON lines\r\n"
    .ascii "  --bench      timings on stderr: load, prefill tok/s, decode tok/s\r\n"
    .ascii "  --threads N  number of threads (default: all logical processors)\r\n"
    .ascii "  --q8         quantize an fp32 (LLMA) checkpoint to int8 at load (faster)\r\n"
    .ascii "  A prompt longer than the context keeps its most recent tokens (with a\r\n"
    .ascii "  warning on stderr); if nothing can be generated it prints {\"error\":...}\r\n"
    .asciz "  and exits nonzero.\r\n"
s_unsup1:       .asciz "llm: option '"
s_unsup2:       .asciz "' is not supported in asm port (only plain single-sample hfgen)"
s_nocpu:        .asciz "llm: error: this CPU (or OS) lacks AVX2+FMA, which the asm port requires"
s_tokfail:      .asciz "hfgen: error: cannot load tokenizer '"
s_quote:        .asciz "'"
s_cannot_open:  .asciz "cannot open '"
s_notckpt1:     .asciz "error: '"
s_notckpt2:     .asciz "' is not a GPTC/LLMA/LLMQ checkpoint"
s_notckpt_json: .asciz "not a GPTC/LLMA/LLMQ checkpoint"
s_gptc:         .asciz "GPT-2 (GPTC) checkpoints are not supported in asm port"
s_cannot_load:  .asciz "cannot load model"
s_oom:          .asciz "out of memory"
s_empty:        .asciz "empty prompt: nothing to generate from"
s_small:        .asciz "model context too small to generate"
s_hferr:        .asciz "hfgen: error: "
s_jerr1:        .asciz "{\"error\":\""
s_jerr2:        .asciz "\"}\r\n"
s_done:         .asciz "{\"done\":true}\r\n"
s_d_open:       .ascii "{\"d\":\""
s_d_close:      .ascii "\"}\r\n"
s_warn1:        .asciz "hfgen: warning: prompt is "
s_warn2:        .asciz " tokens but the context is "
s_warn3:        .asciz "; dropped the oldest "
s_warn4:        .asciz " tokens (kept the last "
s_warn5:        .asciz ", room for "
s_warn6:        .asciz " new)"
s_b_load:       .asciz "bench: threads "
s_b_tok:        .asciz ", tokenizer load+encode "
s_b_model:      .asciz " s, model load "
s_b_s:          .asciz " s"
s_b_prompt:     .asciz "bench: prompt "
s_b_prefill:    .asciz " tokens, prefill "
s_b_eq:         .asciz " s = "
s_b_tps:        .asciz " tok/s"
s_b_decode:     .asciz "bench: decode "
s_b_gen:        .asciz " tokens, "
s_b_total:      .asciz "bench: total "

// hf_stop patterns. End markers that some models print as plain text:
end_1: .ascii "<|endoftext|>"
end_1e:
end_2: .ascii "<|im_end|>"
end_2e:
end_3: .ascii "<|eot_id|>"
end_3e:
    .balign 8
end_tab:                                // (pointer, length) pairs, 0-terminated
    .quad end_1, end_1e - end_1
    .quad end_2, end_2e - end_2
    .quad end_3, end_3e - end_3
    .quad 0, 0
// role labels that start a hallucinated next turn when they begin a line
lab_1: .ascii "User"
lab_2: .ascii "You"
lab_3: .ascii "Human"
lab_4: .ascii "Assistant"
lab_5: .ascii "AI"
lab_6: .ascii "System"
lab_7: .ascii "Q"
lab_e:
    .balign 8
lab_tab:
    .quad lab_1, 4
    .quad lab_2, 3
    .quad lab_3, 5
    .quad lab_4, 9
    .quad lab_5, 2
    .quad lab_6, 6
    .quad lab_7, 1
    .quad 0, 0

    .bss
    .balign 64
g_model:    .space M_SIZE               // the Model
g_kv:       .space KV_SIZE              // its KV cache
g_rng:      .space 64                   // sampler RNG state (sample.S owns the layout)
g_pos:      .space 8 * MAX_POS          // positional arguments (wchar_t*)
g_npos:     .quad 0
g_argv:     .quad 0
g_argc:     .quad 0
g_raw:      .quad 0
g_bench:    .quad 0
g_q8:       .quad 0
g_threads:  .quad 0
g_bad:      .quad 0
g_unsup:    .quad 0                     // first unsupported option seen
g_n_new:    .quad 0
g_temp:     .long 0                     // f32
            .long 0
g_seed:     .quad 0
g_tok:      .quad 0
g_prompt:   .quad 0
g_plen:     .quad 0
g_pids:     .quad 0
g_np:       .quad 0
g_first:    .quad 0
g_keep:     .quad 0
g_logits:   .quad 0
g_dec:      .quad 0                     // decoded text so far (DEC_CAP bytes)
g_piece:    .quad 0                     // one token's bytes
g_line:     .quad 0                     // one JSON output line
g_tvocab:   .quad 0                     // tokenizer vocab size
g_gn:       .quad 0                     // generated (non-EOG) tokens
g_t0:       .quad 0                     // timestamps (ns)
g_t_tok:    .quad 0
g_t_load:   .quad 0
g_t_pf:     .quad 0
g_t_end:    .quad 0

    .text

#ifndef LLM_NO_ENTRY
// ===========================================================================
// Process entry point (no CRT): run the program, exit with its return code.
// ===========================================================================
    .globl mainCRTStartup
mainCRTStartup:
    sub rsp, 40                         // entry rsp is 8 mod 16, like any function
    call llm_main
    mov ecx, eax
    call qword ptr [rip + __imp_ExitProcess]
    int3
#endif

// ---------------------------------------------------------------------------
// i64 wcs_eq(const wchar_t* w, const char* a): 1 if the UTF-16 string equals the
// ASCII string. Leaf: clobbers rax r8.
// ---------------------------------------------------------------------------
wcs_eq:
1:  movzx eax, word ptr [rcx]
    movzx r8d, byte ptr [rdx]
    cmp eax, r8d
    jne 2f
    test eax, eax
    jz 3f
    add rcx, 2
    inc rdx
    jmp 1b
2:  xor eax, eax
    ret
3:  mov eax, 1
    ret

// ---------------------------------------------------------------------------
// emit_error(const char* msg): stderr "hfgen: error: msg"; JSON {"error":"msg"}
// unless --raw (gen.c emit_error; the message is printed unescaped, as in C)
// ---------------------------------------------------------------------------
emit_error:
    push rbx
    sub rsp, 32
    mov rbx, rcx
    lea rcx, [rip + s_hferr]
    call msg_z
    mov rcx, rbx
    call msg_z
    call msg_nl
    call msg_to_err
    cmp qword ptr [rip + g_raw], 0
    jne 1f
    lea rcx, [rip + s_jerr1]
    call io_out_z
    mov rcx, rbx
    call io_out_z
    lea rcx, [rip + s_jerr2]
    call io_out_z
    call io_flush
1:  add rsp, 32
    pop rbx
    ret

// emit_done(): {"done":true} unless --raw
emit_done:
    sub rsp, 40
    cmp qword ptr [rip + g_raw], 0
    jne 1f
    lea rcx, [rip + s_done]
    call io_out_z
1:  call io_flush
    add rsp, 40
    ret

// ---------------------------------------------------------------------------
// i64 hf_scan(const u8* s, i64 from, i64 end) -> cut position or -1
// gen.c hf_stop over the text s[0, end): the EARLIEST position p >= from where
//   - an end marker (<|endoftext|>, <|im_end|>, <|eot_id|>) starts, or
//   - a '\n' is followed by a role label (User You Human Assistant AI System Q)
//     that is a whole word: followed by end of text, ':', ' ', '\t', '\r' or '\n'.
// Since hf_stop ran (and found nothing) after every earlier token, only matches
// touching the new bytes are possible: the caller starts at old_end - 15
// (longest pattern: 13-byte "<|endoftext|>"; "\nAssistant" + 1 = 11 bytes).
// ---------------------------------------------------------------------------
hf_scan:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    mov rsi, rcx                        // rsi = text
    mov r12, rdx                        // r12 = p
    mov r13, r8                         // r13 = end
.Lhs_pos:
    cmp r12, r13
    jae .Lhs_none
    movzx eax, byte ptr [rsi + r12]
    cmp eax, '<'
    je .Lhs_ends
    cmp eax, 10
    je .Lhs_labels
.Lhs_next:
    inc r12
    jmp .Lhs_pos
.Lhs_ends:
    lea rbx, [rip + end_tab]
.Lhs_end_loop:
    mov rdi, qword ptr [rbx]            // pattern
    test rdi, rdi
    jz .Lhs_next
    mov rcx, qword ptr [rbx + 8]        // its length
    lea rax, [r12 + rcx]
    cmp rax, r13
    ja 2f                               // does not fit before the end
    lea rdx, [rsi + r12]
    xor eax, eax
1:  movzx r8d, byte ptr [rdx + rax]
    cmp r8b, byte ptr [rdi + rax]
    jne 2f
    inc rax
    cmp rax, rcx
    jb 1b
    jmp .Lhs_hit                        // all bytes equal
2:  add rbx, 16
    jmp .Lhs_end_loop
.Lhs_labels:
    lea rbx, [rip + lab_tab]
.Lhs_lab_loop:
    mov rdi, qword ptr [rbx]
    test rdi, rdi
    jz .Lhs_next
    mov rcx, qword ptr [rbx + 8]
    lea rax, [r12 + 1 + rcx]            // index just after the label
    cmp rax, r13
    ja 4f
    lea rdx, [rsi + r12 + 1]
    xor eax, eax
3:  movzx r8d, byte ptr [rdx + rax]
    cmp r8b, byte ptr [rdi + rax]
    jne 4f
    inc rax
    cmp rax, rcx
    jb 3b
    // label matched: check the character after it
    lea rax, [r12 + 1 + rcx]
    cmp rax, r13
    je .Lhs_hit                         // end of text ('\0' in C)
    movzx eax, byte ptr [rsi + rax]
    cmp eax, ':'
    je .Lhs_hit
    cmp eax, ' '
    je .Lhs_hit
    cmp eax, 9
    je .Lhs_hit
    cmp eax, 13
    je .Lhs_hit
    cmp eax, 10
    je .Lhs_hit
    jmp .Lhs_next                       // C breaks out of the label loop here too
4:  add rbx, 16
    jmp .Lhs_lab_loop
.Lhs_hit:
    mov rax, r12
    jmp .Lhs_ret
.Lhs_none:
    mov rax, -1
.Lhs_ret:
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

// ---------------------------------------------------------------------------
// i64 generate(void) -> 0 ok, 1 nothing could be generated (gen.c dec_generate)
// ---------------------------------------------------------------------------
generate:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    push r14
    push r15
    sub rsp, 64                         // 8 + 56 + 64 = 128; [rsp+32],[rsp+40] = args 5,6
    lea rbx, [rip + g_model]            // rbx = Model*

    // ---- hfgen_fit_prompt(np, max_seq, n_new, &first) ----
    mov rax, qword ptr [rip + g_np]     // rax = np
    mov rcx, qword ptr [rbx + M_MAXSEQ] // rcx = max_seq
    mov qword ptr [rip + g_first], 0
    xor r12d, r12d                      // r12 = keep
    test rax, rax
    jle .Lgn_fit_done
    cmp rcx, 2
    jl .Lgn_fit_done
    mov rdx, qword ptr [rip + g_n_new]  // reserve = n_new
    mov r8, rcx
    shr r8, 1                           // max_seq / 2
    cmp rdx, r8
    jle 1f
    mov rdx, r8
1:  cmp rdx, 1
    jge 2f
    mov edx, 1
2:  lea r8, [rax + rdx]                 // np + reserve <= max_seq: keep all
    mov r12, rax
    cmp r8, rcx
    jle .Lgn_fit_done
    mov r12, rcx
    sub r12, rdx                        // keep = max_seq - reserve
    mov r8, rax
    sub r8, r12
    mov qword ptr [rip + g_first], r8   // first = np - keep
.Lgn_fit_done:
    mov qword ptr [rip + g_keep], r12
    test r12, r12
    jnz 3f
    lea rcx, [rip + s_empty]
    cmp qword ptr [rip + g_np], 0
    jle 4f
    lea rcx, [rip + s_small]
4:  call emit_error
    mov eax, 1
    jmp .Lgn_ret
3:  cmp qword ptr [rip + g_first], 0
    jle 5f
    // "hfgen: warning: prompt is %d tokens but the context is %d; dropped the
    //  oldest %d tokens (kept the last %d, room for %d new)"
    lea rcx, [rip + s_warn1]
    call msg_z
    mov rcx, qword ptr [rip + g_np]
    call msg_i64
    lea rcx, [rip + s_warn2]
    call msg_z
    mov rcx, qword ptr [rbx + M_MAXSEQ]
    call msg_i64
    lea rcx, [rip + s_warn3]
    call msg_z
    mov rcx, qword ptr [rip + g_first]
    call msg_i64
    lea rcx, [rip + s_warn4]
    call msg_z
    mov rcx, qword ptr [rip + g_keep]
    call msg_i64
    lea rcx, [rip + s_warn5]
    call msg_z
    mov rcx, qword ptr [rbx + M_MAXSEQ]
    sub rcx, qword ptr [rip + g_keep]
    call msg_i64
    lea rcx, [rip + s_warn6]
    call msg_z
    call msg_nl
    call msg_to_err
5:
    // ---- KV cache just big enough: keep + n_new positions (clamped to max_seq) ----
    mov r8, qword ptr [rip + g_n_new]
    test r8, r8
    jns 6f
    xor r8d, r8d
6:  add r8, qword ptr [rip + g_keep]
    mov rcx, rbx
    lea rdx, [rip + g_kv]
    call kv_create_len
    test rax, rax
    jnz .Lgn_oom
    // ---- buffers ----
    mov rcx, qword ptr [rbx + M_VOCAB]
    shl rcx, 2
    call mem_alloc
    mov qword ptr [rip + g_logits], rax
    test rax, rax
    jz .Lgn_oom
    mov ecx, DEC_CAP + 64
    call mem_alloc
    mov qword ptr [rip + g_dec], rax
    test rax, rax
    jz .Lgn_oom
    mov rcx, qword ptr [rip + g_tok]
    call tok_max_piece
    lea r13, [rax + 64]                 // r13 = piece buffer size
    mov rcx, r13
    call mem_alloc
    mov qword ptr [rip + g_piece], rax
    test rax, rax
    jz .Lgn_oom
    lea rcx, [r13 + r13*2]              // JSON line: 6 bytes per input byte + frame
    add rcx, rcx
    add rcx, 64
    call mem_alloc
    mov qword ptr [rip + g_line], rax
    test rax, rax
    jz .Lgn_oom
    mov rcx, qword ptr [rip + g_tok]
    call tok_vocab
    mov qword ptr [rip + g_tvocab], rax

    // ---- prefill: all but the last kept token in a batch, then the last one
    //      through decode_step to get its logits ----
    call io_time_ns
    mov qword ptr [rip + g_t_load], rax     // (end of load phase = start of prefill)
    mov rsi, qword ptr [rip + g_pids]
    mov rax, qword ptr [rip + g_first]
    lea rsi, [rsi + rax*4]              // rsi = first kept id
    mov r9, qword ptr [rip + g_keep]
    dec r9
    jz 7f
    mov rcx, rbx
    lea rdx, [rip + g_kv]
    mov r8, rsi
    call prefill
7:  mov rax, qword ptr [rip + g_keep]
    movsxd r8, dword ptr [rsi + rax*4 - 4]
    mov rcx, rbx
    lea rdx, [rip + g_kv]
    mov r9, qword ptr [rip + g_logits]
    call decode_step
    call io_time_ns
    mov qword ptr [rip + g_t_pf], rax

    // ---- sampling loop ----
    lea rcx, [rip + g_rng]
    mov rdx, qword ptr [rip + g_seed]
    call rng_seed
    mov qword ptr [rip + g_gn], 0
    mov r12, qword ptr [rip + g_keep]   // r12 = pos
    xor r13d, r13d                      // r13 = s (step)
    xor r14d, r14d                      // r14 = prevlen (bytes already printed)
    xor r15d, r15d                      // r15 = efflen (strlen of the decoded text)
    xor edi, edi                        // rdi = declen (bytes in g_dec, cap DEC_CAP-1)
    mov qword ptr [rsp + 56], 0         // [rsp+56] = a NUL byte was decoded
.Lgn_loop:
    cmp r13, qword ptr [rip + g_n_new]
    jge .Lgn_end
    cmp r12, qword ptr [rbx + M_MAXSEQ]
    jge .Lgn_end
    // nx = sample_token(logits, vocab, temp, top_k, top_p, &rng)
    mov rcx, qword ptr [rip + g_logits]
    mov rdx, qword ptr [rbx + M_VOCAB]
    movss xmm2, dword ptr [rip + g_temp]
    mov r9d, TOP_K
    mov eax, dword ptr [rip + f_top_p]
    mov qword ptr [rsp + 32], rax       // float in the low half of the stack slot
    lea rax, [rip + g_rng]
    mov qword ptr [rsp + 40], rax
    call sample_token
    mov rsi, rax                        // rsi = nx
    mov rcx, qword ptr [rip + g_tok]
    mov rdx, rsi
    call tok_is_eog
    test rax, rax
    jnz .Lgn_end                        // end of generation: not printed
    inc qword ptr [rip + g_gn]
    // ---- append the token's bytes (hftok_decode skips ids outside the vocab) ----
    mov qword ptr [rsp + 48], r15       // [rsp+48] = old efflen
    test rsi, rsi
    js .Lgn_scan
    cmp rsi, qword ptr [rip + g_tvocab]
    jge .Lgn_scan
    mov rcx, qword ptr [rip + g_tok]
    mov rdx, rsi
    mov r8, qword ptr [rip + g_piece]
    call tok_piece                      // rax = piece length
    mov rcx, DEC_CAP - 1
    sub rcx, rdi                        // room left in the decode buffer
    cmp rax, rcx
    jle 8f
    mov rax, rcx
8:  test rax, rax
    jle .Lgn_scan
    // copy piece -> dec + declen, watching for a NUL (C strlen semantics)
    mov r8, qword ptr [rip + g_piece]
    mov r9, qword ptr [rip + g_dec]
    add r9, rdi
    xor ecx, ecx
9:  movzx edx, byte ptr [r8 + rcx]
    mov byte ptr [r9 + rcx], dl
    test edx, edx
    jnz 10f
    cmp qword ptr [rsp + 56], 0
    jne 10f
    mov qword ptr [rsp + 56], 1         // first NUL: the visible text ends here
    lea r15, [rdi + rcx]
10: inc rcx
    cmp rcx, rax
    jb 9b
    add rdi, rax
    cmp qword ptr [rsp + 56], 0
    jne .Lgn_scan
    mov r15, rdi                        // no NUL: everything is visible
.Lgn_scan:
    // ---- hf_stop on the new tail ----
    mov rdx, qword ptr [rsp + 48]
    sub rdx, 15
    jns 11f
    xor edx, edx
11: mov rcx, qword ptr [rip + g_dec]
    mov r8, r15
    call hf_scan
    xor r8d, r8d                        // r8 = stop?
    test rax, rax
    js 12f
    mov r15, rax                        // cut the text there
    mov r8d, 1
12: mov qword ptr [rsp + 48], r8
    // ---- print bytes [prevlen, efflen) ----
    cmp qword ptr [rip + g_raw], 0
    jne .Lgn_raw
    mov rcx, qword ptr [rip + g_line]   // {"d":" + escaped + "}\r\n
    mov eax, dword ptr [rip + s_d_open]
    mov dword ptr [rcx], eax
    mov ax, word ptr [rip + s_d_open + 4]
    mov word ptr [rcx + 4], ax
    add rcx, 6
    mov rdx, qword ptr [rip + g_dec]
    add rdx, r14
    mov r8, r15
    sub r8, r14                         // may be <= 0: then nothing
    call json_escape
    mov rcx, qword ptr [rip + g_line]
    lea rdx, [rcx + rax + 6]
    mov eax, dword ptr [rip + s_d_close]
    mov dword ptr [rdx], eax
    add rdx, 4
    sub rdx, rcx                        // line length
    call io_out
    jmp .Lgn_flush
.Lgn_raw:
    mov rdx, r15
    sub rdx, r14
    jle .Lgn_flush
    mov rcx, qword ptr [rip + g_dec]
    add rcx, r14
    call io_out
.Lgn_flush:
    call io_flush                       // stream: every token goes out now
    mov r14, r15                        // prevlen = efflen
    cmp qword ptr [rsp + 48], 0
    jne .Lgn_end                        // stopped: the token is not decoded
    // ---- feed the token back ----
    mov rcx, rbx
    lea rdx, [rip + g_kv]
    mov r8, rsi
    mov r9, qword ptr [rip + g_logits]
    call decode_step
    inc r12
    inc r13
    jmp .Lgn_loop
.Lgn_end:
    call io_time_ns
    mov qword ptr [rip + g_t_end], rax
    xor eax, eax
    jmp .Lgn_ret
.Lgn_oom:
    lea rcx, [rip + s_oom]
    call emit_error
    mov eax, 1
.Lgn_ret:
    add rsp, 64
    pop r15
    pop r14
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

// msg_secs(i64 ns): append ns as seconds with 3 decimals
msg_secs:
    sub rsp, 40
    cvtsi2sd xmm0, rcx
    divsd xmm0, qword ptr [rip + d_1e9]
    mov edx, 3
    call msg_f64
    add rsp, 40
    ret

// msg_rate(i64 count, i64 ns): append count / seconds with 1 decimal
msg_rate:
    sub rsp, 40
    test rdx, rdx
    jg 1f
    mov edx, 1
1:  cvtsi2sd xmm0, rcx
    cvtsi2sd xmm1, rdx
    mulsd xmm0, qword ptr [rip + d_1e9]
    divsd xmm0, xmm1
    mov edx, 1
    call msg_f64
    add rsp, 40
    ret

// bench_report(): timings on stderr
bench_report:
    sub rsp, 40
    lea rcx, [rip + s_b_load]           // bench: threads T, tokenizer load+encode X s, model load Y s
    call msg_z
    call pool_threads
    mov rcx, rax
    call msg_i64
    lea rcx, [rip + s_b_tok]
    call msg_z
    mov rcx, qword ptr [rip + g_t_tok]
    sub rcx, qword ptr [rip + g_t0]
    call msg_secs
    lea rcx, [rip + s_b_model]
    call msg_z
    mov rcx, qword ptr [rip + g_t_load]
    sub rcx, qword ptr [rip + g_t_tok]
    call msg_secs
    lea rcx, [rip + s_b_s]
    call msg_z
    call msg_nl
    lea rcx, [rip + s_b_prompt]         // bench: prompt N tokens, prefill X s = R tok/s
    call msg_z
    mov rcx, qword ptr [rip + g_keep]
    call msg_i64
    lea rcx, [rip + s_b_prefill]
    call msg_z
    mov rcx, qword ptr [rip + g_t_pf]
    sub rcx, qword ptr [rip + g_t_load]
    call msg_secs
    lea rcx, [rip + s_b_eq]
    call msg_z
    mov rcx, qword ptr [rip + g_keep]
    mov rdx, qword ptr [rip + g_t_pf]
    sub rdx, qword ptr [rip + g_t_load]
    call msg_rate
    lea rcx, [rip + s_b_tps]
    call msg_z
    call msg_nl
    lea rcx, [rip + s_b_decode]         // bench: decode N tokens, X s = R tok/s
    call msg_z
    mov rcx, qword ptr [rip + g_gn]
    call msg_i64
    lea rcx, [rip + s_b_gen]
    call msg_z
    mov rcx, qword ptr [rip + g_t_end]
    sub rcx, qword ptr [rip + g_t_pf]
    call msg_secs
    lea rcx, [rip + s_b_eq]
    call msg_z
    mov rcx, qword ptr [rip + g_gn]
    mov rdx, qword ptr [rip + g_t_end]
    sub rdx, qword ptr [rip + g_t_pf]
    call msg_rate
    lea rcx, [rip + s_b_tps]
    call msg_z
    call msg_nl
    lea rcx, [rip + s_b_total]          // bench: total X s
    call msg_z
    mov rcx, qword ptr [rip + g_t_end]
    sub rcx, qword ptr [rip + g_t0]
    call msg_secs
    lea rcx, [rip + s_b_s]
    call msg_z
    call msg_nl
    call msg_to_err
    add rsp, 40
    ret

// ===========================================================================
// i64 llm_main(void) -> process exit code (0 ok, 1 error, 2 usage)
// ===========================================================================
    .globl llm_main
llm_main:
    push rbx
    push rsi
    push rdi
    push r12
    sub rsp, 56                         // 8 + 32 + 56 = 96: aligned for calls
    call io_time_ns
    mov qword ptr [rip + g_t0], rax

    // ---- argv (UTF-16) from the command line ----
    call qword ptr [rip + __imp_GetCommandLineW]
    mov rcx, rax
    lea rdx, [rip + g_argc]             // CommandLineToArgvW writes an int
    mov qword ptr [rdx], 0
    call qword ptr [rip + __imp_CommandLineToArgvW]
    mov qword ptr [rip + g_argv], rax
    test rax, rax
    jz .Lmn_usage
    movsxd rax, dword ptr [rip + g_argc]
    mov qword ptr [rip + g_argc], rax

    // ---- options (anywhere after argv[0]); the rest are positional ----
    mov ebx, 1                          // rbx = i
.Lmn_arg:
    cmp rbx, qword ptr [rip + g_argc]
    jge .Lmn_args_done
    mov rax, qword ptr [rip + g_argv]
    mov rsi, qword ptr [rax + rbx*8]    // rsi = argv[i]
    mov rcx, rsi
    lea rdx, [rip + s_raw]
    call wcs_eq
    test eax, eax
    jz 1f
    mov qword ptr [rip + g_raw], 1
    jmp .Lmn_arg_next
1:  mov rcx, rsi
    lea rdx, [rip + s_bench]
    call wcs_eq
    test eax, eax
    jz 2f
    mov qword ptr [rip + g_bench], 1
    jmp .Lmn_arg_next
2:  mov rcx, rsi
    lea rdx, [rip + s_q8]
    call wcs_eq
    test eax, eax
    jz 3f
    mov qword ptr [rip + g_q8], 1
    jmp .Lmn_arg_next
3:  mov rcx, rsi
    lea rdx, [rip + s_threads]
    call wcs_eq
    test eax, eax
    jz 4f
    lea rax, [rbx + 1]
    cmp rax, qword ptr [rip + g_argc]
    jge .Lmn_bad
    mov rbx, rax
    mov rax, qword ptr [rip + g_argv]
    mov rcx, qword ptr [rax + rbx*8]
    call parse_i64
    mov qword ptr [rip + g_threads], rax
    jmp .Lmn_arg_next
4:  // options of transformer hfgen that this port does not implement:
    // remember the first one (reported after the loop)
    lea rdi, [rip + unsup_tab]
41: mov rdx, qword ptr [rdi]
    test rdx, rdx
    jz 42f
    mov rcx, rsi
    call wcs_eq                         // (leaves rdi alone)
    test eax, eax
    jnz 43f
    add rdi, 8
    jmp 41b
43: cmp qword ptr [rip + g_unsup], 0
    jne .Lmn_arg_next
    mov qword ptr [rip + g_unsup], rsi
    jmp .Lmn_arg_next
42: // any other "--x" is an error; else a positional argument
    cmp word ptr [rsi], '-'
    jne 5f
    cmp word ptr [rsi + 2], '-'
    jne 5f
.Lmn_bad:
    mov qword ptr [rip + g_bad], 1
    jmp .Lmn_arg_next
5:  mov rax, qword ptr [rip + g_npos]
    cmp rax, MAX_POS
    jge .Lmn_arg_next
    lea rcx, [rip + g_pos]
    mov qword ptr [rcx + rax*8], rsi
    inc qword ptr [rip + g_npos]
.Lmn_arg_next:
    inc rbx
    jmp .Lmn_arg

.Lmn_args_done:
    mov rax, qword ptr [rip + g_unsup]
    test rax, rax
    jz 7f
    lea rcx, [rip + s_unsup1]
    call msg_z
    mov rcx, qword ptr [rip + g_unsup]
    call msg_w
    lea rcx, [rip + s_unsup2]
    call msg_z
    call msg_nl
    call msg_to_err
    mov eax, 2
    jmp .Lmn_ret
7:  cmp qword ptr [rip + g_bad], 0
    jne .Lmn_usage
    cmp qword ptr [rip + g_npos], 2
    jge .Lmn_parse

.Lmn_usage:
    lea rcx, [rip + s_usage]
    call io_err_z
    mov eax, 2
    jmp .Lmn_ret

.Lmn_parse:
    lea r12, [rip + g_pos]
    // n_new = atoi(argv) (an int: the low 32 bits, sign-extended)
    mov qword ptr [rip + g_n_new], 128
    cmp qword ptr [rip + g_npos], 2
    jle 8f
    mov rcx, qword ptr [r12 + 16]
    call parse_i64
    movsxd rax, eax
    mov qword ptr [rip + g_n_new], rax
8:  // temp = (float)atof(argv)
    mov eax, dword ptr [rip + f_temp_def]
    mov dword ptr [rip + g_temp], eax
    cmp qword ptr [rip + g_npos], 3
    jle 9f
    mov rcx, qword ptr [r12 + 24]
    call parse_f64
    cvtsd2ss xmm0, xmm0
    movss dword ptr [rip + g_temp], xmm0
9:  // seed = strtoull(argv)
    mov qword ptr [rip + g_seed], 1234
    cmp qword ptr [rip + g_npos], 4
    jle 10f
    mov rcx, qword ptr [r12 + 32]
    call parse_u64
    mov qword ptr [rip + g_seed], rax
10:
    // ---- CPU features + thread pool ----
    call cpu_init
    test eax, 1
    jnz 11f
    lea rcx, [rip + s_nocpu]
    call msg_z
    call msg_nl
    call msg_to_err
    mov eax, 1
    jmp .Lmn_ret
11: mov rcx, qword ptr [rip + g_threads]
    call pool_init

    // ---- tokenizer ----
    mov rcx, qword ptr [r12 + 8]
    call tok_load
    mov qword ptr [rip + g_tok], rax
    test rax, rax
    jnz 12f
    lea rcx, [rip + s_tokfail]
    call msg_z
    mov rcx, qword ptr [r12 + 8]
    call msg_w
    lea rcx, [rip + s_quote]
    call msg_z
    call msg_nl
    call msg_to_err
    mov eax, 1
    jmp .Lmn_ret
12:
    // ---- prompt: all of stdin, as the C runtime would deliver it ----
    lea rcx, [rip + g_plen]
    call io_read_stdin
    mov qword ptr [rip + g_prompt], rax
    test rax, rax
    jz .Lmn_oom_plain
    mov rcx, rax
    mov rdx, qword ptr [rip + g_plen]
    call io_crt_text
    // C passes the prompt as a C string: it ends at the first NUL
    mov rcx, qword ptr [rip + g_prompt]
    xor edx, edx
13: cmp rdx, rax
    jae 14f
    cmp byte ptr [rcx + rdx], 0
    je 14f
    inc rdx
    jmp 13b
14: mov qword ptr [rip + g_plen], rdx
    // ids: plen + 16 slots, as main.c sizes them
    lea rcx, [rdx + 16]
    shl rcx, 2
    call mem_alloc
    mov qword ptr [rip + g_pids], rax
    test rax, rax
    jz .Lmn_oom_plain
    mov rcx, qword ptr [rip + g_tok]    // np = tok_encode(tok, text, len, ids, cap)
    mov rdx, qword ptr [rip + g_prompt]
    mov r8, qword ptr [rip + g_plen]
    mov r9, rax
    lea rax, [r8 + 16]
    mov qword ptr [rsp + 32], rax
    call tok_encode
    mov qword ptr [rip + g_np], rax
    call io_time_ns
    mov qword ptr [rip + g_t_tok], rax

    // ---- hfgen_file: dispatch on the magic ----
    mov rcx, qword ptr [r12]
    call model_peek_magic
    cmp rax, -1
    jne 15f
    lea rcx, [rip + s_cannot_open]      // "cannot open '<path>'" and nothing on stdout
    call msg_z
    mov rcx, qword ptr [r12]
    call msg_w
    lea rcx, [rip + s_quote]
    call msg_z
    call msg_nl
    call msg_to_err
    mov eax, 1
    jmp .Lmn_ret
15: cmp eax, GPTC_MAGIC
    jne 16f
    lea rcx, [rip + s_gptc]
    call emit_error
    jmp .Lmn_fail_done
16: cmp eax, LLMA_MAGIC
    je 17f
    cmp eax, LLMQ_MAGIC
    je 17f
    lea rcx, [rip + s_notckpt1]         // "error: '<path>' is not a GPTC/LLMA/LLMQ checkpoint"
    call msg_z
    mov rcx, qword ptr [r12]
    call msg_w
    lea rcx, [rip + s_notckpt2]
    call msg_z
    call msg_nl
    call msg_to_err
    cmp qword ptr [rip + g_raw], 0
    jne .Lmn_fail_done
    lea rcx, [rip + s_jerr1]
    call io_out_z
    lea rcx, [rip + s_notckpt_json]
    call io_out_z
    lea rcx, [rip + s_jerr2]
    call io_out_z
    jmp .Lmn_fail_done
17: mov rcx, qword ptr [r12]            // model_load(path, &model, q8 ? 1 : 0)
    lea rdx, [rip + g_model]
    mov r8, qword ptr [rip + g_q8]
    call model_load
    test rax, rax
    jz 18f
    lea rcx, [rip + s_cannot_load]
    call emit_error
    jmp .Lmn_fail_done
18: call generate
    mov rsi, rax                        // rsi = return code
    call emit_done
    cmp qword ptr [rip + g_bench], 0
    je 19f
    test rsi, rsi
    jnz 19f
    call bench_report
19: mov rax, rsi
    jmp .Lmn_ret

.Lmn_fail_done:
    call emit_done
    mov eax, 1
    jmp .Lmn_ret
.Lmn_oom_plain:
    lea rcx, [rip + s_oom]
    call emit_error
    mov eax, 1
.Lmn_ret:
    mov rbx, rax
    call io_flush
    mov rax, rbx
    add rsp, 56
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret
