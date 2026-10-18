// chat.S - the `hfchat` conversation format and chat templates (owner: CLI agent)
//
// Port of ../main.c run_hfchat's turn parser and ../hftok.c hftok_chat_render.
//
// Exported (Microsoft x64 ABI, ints are 64-bit):
//   i64 chat_parse(u8* in, const u8** roles, const u8** contents, i64 max_turns)
//         -> number of turns
//       `in` is the NUL-terminated conversation from stdin: turns separated by
//       0x1E ("record separator"), each turn "role" 0x1F ("unit separator")
//       "content". Parsed IN PLACE like the C code: every separator becomes a NUL,
//       so roles[i] and contents[i] are C strings pointing into `in`. A record
//       without 0x1F is skipped; parsing stops at max_turns (256 in C) turns.
//   i64 chat_render(Tok* t, const u8** roles, const u8** contents, i64 nturns,
//                   u8* out, i64 max_out) -> length (out is NUL-terminated)
//       Renders the conversation in the model's own chat template (the kind is
//       stored in the .tok file, see tok.S TK_TMPL), followed by the opening of
//       the assistant's turn, so the model continues as the assistant:
//
//       kind 2 (LLAMA3):  <|begin_of_text|>
//                         [<|start_header_id|>system<|end_header_id|>\n\n SYS <|eot_id|>]
//                         per turn: <|start_header_id|>ROLE<|end_header_id|>\n\n TEXT <|eot_id|>
//                         <|start_header_id|>assistant<|end_header_id|>\n\n
//       kind 1 (ChatML):  [<|im_start|>system\n SYS <|im_end|>\n]
//                         per turn: <|im_start|>ROLE\n TEXT <|im_end|>\n
//                         <|im_start|>assistant\n
//       else (plain):     per turn: "User: " / "Assistant: " / "System: " TEXT \n
//                         Assistant:
//
//       SYS = the model's default system prompt, inserted only when no turn has
//       the role "system" and the prompt is not empty. Like the C sb_put, output
//       stops silently at max_out - 1 bytes.
//
// Byte-identical to the C code by construction: the same strings in the same
// order, the same truncation rule.
//
// ---------------------------------------------------------------------------
// Asm notes for the reader
//   * The internal helper `put` uses a private register convention (documented
//     at it) instead of the Win64 ABI: it is a leaf (calls nothing) and only
//     called from chat_render, so it can keep the output state in callee-saved
//     registers that chat_render owns. That saves loading/storing the position
//     for every string.
//   * Strings in .rdata are NUL-terminated (.asciz) because `put` copies up to
//     the first NUL, exactly like the C loop `while (*s && ...)`.
// ---------------------------------------------------------------------------

    .intel_syntax noprefix

// Tok fields we read (tok.S owns the layout; see the table at its top)
#define TK_TMPL      32     // i64 chat template kind (HFTOK_TMPL_*)
#define TK_SYSPTR    80     // u8* default system prompt (no NUL; may be 0)
#define TK_SYSLEN    88     // i64 its length

#define TMPL_CHATML  1
#define TMPL_LLAMA3  2

    .section .rdata, "dr"
s_system:     .asciz "system"
s_user:       .asciz "user"
s_assistant:  .asciz "assistant"
// LLAMA3
l3_bos:       .asciz "<|begin_of_text|>"
l3_sys:       .asciz "<|start_header_id|>system<|end_header_id|>\n\n"
l3_eot:       .asciz "<|eot_id|>"
l3_hdr:       .asciz "<|start_header_id|>"
l3_hdr_end:   .asciz "<|end_header_id|>\n\n"
l3_asst:      .asciz "<|start_header_id|>assistant<|end_header_id|>\n\n"
// ChatML
cm_sys:       .asciz "<|im_start|>system\n"
cm_end:       .asciz "<|im_end|>\n"
cm_start:     .asciz "<|im_start|>"
cm_nl:        .asciz "\n"
cm_asst:      .asciz "<|im_start|>assistant\n"
// plain
pl_user:      .asciz "User: "
pl_asst:      .asciz "Assistant: "
pl_sys:       .asciz "System: "
pl_end:       .asciz "Assistant:"

    .text

// ---------------------------------------------------------------------------
// i64 str_eq(const u8* a, const u8* b): strcmp(a, b) == 0. Leaf; clobbers rax r8.
// ---------------------------------------------------------------------------
str_eq:
1:  movzx eax, byte ptr [rcx]
    movzx r8d, byte ptr [rdx]
    cmp eax, r8d
    jne 2f
    test eax, eax
    jz 3f                               // both ended together: equal
    inc rcx
    inc rdx
    jmp 1b
2:  xor eax, eax
    ret
3:  mov eax, 1
    ret

// ---------------------------------------------------------------------------
// u8* str_chr(u8* s, i64 c): strchr for c != 0 -> pointer or 0. Leaf; rax r8.
// ---------------------------------------------------------------------------
str_chr:
1:  movzx eax, byte ptr [rcx]
    cmp eax, edx
    je 2f
    test eax, eax
    jz 3f
    inc rcx
    jmp 1b
2:  mov rax, rcx
    ret
3:  xor eax, eax
    ret

// ===========================================================================
// i64 chat_parse(u8* in, const u8** roles, const u8** contents, i64 max_turns)
//   C (main.c run_hfchat):
//     while (*p && nt < 256) {
//       char *rec_end = strchr(p, '\x1e'); if (rec_end) *rec_end = '\0';
//       char *us = strchr(p, '\x1f');
//       if (us) { *us = '\0'; roles[nt] = p; contents[nt] = us + 1; nt++; }
//       if (!rec_end) break;
//       p = rec_end + 1;
//     }
// ===========================================================================
    .globl chat_parse
chat_parse:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    mov rsi, rcx                        // rsi = p
    mov r12, rdx                        // r12 = roles
    mov r13, r8                         // r13 = contents
    mov rdi, r9                         // rdi = max_turns
    xor ebx, ebx                        // rbx = nt
.Lcp_loop:
    cmp byte ptr [rsi], 0
    je .Lcp_done
    cmp rbx, rdi
    jge .Lcp_done
    mov rcx, rsi
    mov edx, 0x1E
    call str_chr
    mov r9, rax                         // r9 = rec_end (or 0)
    test rax, rax
    jz 1f
    mov byte ptr [rax], 0               // end this record's string here
1:  mov rcx, rsi
    mov edx, 0x1F
    call str_chr                        // (leaf: keeps r9)
    test rax, rax
    jz 2f
    mov byte ptr [rax], 0               // role ends at the unit separator
    mov qword ptr [r12 + rbx*8], rsi
    inc rax
    mov qword ptr [r13 + rbx*8], rax
    inc rbx
2:  test r9, r9
    jz .Lcp_done                        // that was the last record
    lea rsi, [r9 + 1]
    jmp .Lcp_loop
.Lcp_done:
    mov rax, rbx
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

// ---------------------------------------------------------------------------
// put: append a string to the output, C sb_put semantics.
//   in:  rdx = source, r8 = maximum number of bytes to take from it (the copy
//        also stops at a NUL in the source, and when the output is full)
//   state (owned by chat_render): rbx = out, rsi = position, rdi = max_out - 1
//   clobbers rax, rdx, r8
// ---------------------------------------------------------------------------
put:
1:  test r8, r8
    jz 2f                               // took the maximum
    cmp rsi, rdi
    jge 2f                              // output full (leaves room for the NUL)
    movzx eax, byte ptr [rdx]
    test eax, eax
    jz 2f                               // end of the C string
    mov byte ptr [rbx + rsi], al
    inc rsi
    inc rdx
    dec r8
    jmp 1b
2:  ret

// put_z: put a NUL-terminated string (rdx); r8 = "no limit"
#define PUT_Z(sym)  lea rdx, [rip + sym]; mov r8, -1; call put
#define PUT_R(reg)  mov rdx, reg; mov r8, -1; call put

// ===========================================================================
// i64 chat_render(Tok* t, const u8** roles, const u8** contents, i64 nturns,
//                 u8* out, i64 max_out)
// ===========================================================================
    .globl chat_render
chat_render:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    push r14
    push r15
    push rbp
    sub rsp, 40                         // 8 + 64 + 40 = 112: aligned (only leaf calls anyway)
    mov r12, rcx                        // r12 = Tok*
    mov r13, rdx                        // r13 = roles
    mov r14, r8                         // r14 = contents
    mov r15, r9                         // r15 = nturns
    mov rbx, qword ptr [rsp + 104 + 40] // rbx = out (5th arg: [entry rsp + 40]; entry rsp = rsp + 64 pushed + 40)
    mov rdi, qword ptr [rsp + 104 + 48] // rdi = max_out (6th arg)
    dec rdi                             // keep one byte for the NUL
    xor esi, esi                        // rsi = p = 0

    // ---- has_system: does any turn have the role "system"? ----
    xor ebp, ebp                        // rbp = has_system
    xor r9d, r9d
1:  cmp r9, r15
    jge 2f
    mov rcx, qword ptr [r13 + r9*8]
    lea rdx, [rip + s_system]
    call str_eq                         // (leaf: keeps r9)
    or ebp, eax
    inc r9
    jmp 1b
2:  // ---- does the default system prompt get inserted? -> rbp = 1 if yes ----
    // C: !has_system && t->default_system && t->default_system[0]
    xor eax, eax
    test ebp, ebp
    jnz 3f
    mov rcx, qword ptr [r12 + TK_SYSPTR]
    test rcx, rcx
    jz 3f
    cmp qword ptr [r12 + TK_SYSLEN], 0
    jle 3f                              // empty string ("" in C)
    cmp byte ptr [rcx], 0
    je 3f
    mov eax, 1
3:  mov ebp, eax                        // rbp = insert the default system prompt

    mov rax, qword ptr [r12 + TK_TMPL]
    cmp rax, TMPL_LLAMA3
    je .Lcr_llama3
    cmp rax, TMPL_CHATML
    je .Lcr_chatml

    // ---- plain "User:/Assistant:" fallback ----
    xor r9d, r9d                        // r9 = i (put/str_eq leave r9 alone)
.Lcr_pl_turn:
    cmp r9, r15
    jge .Lcr_pl_end
    mov rcx, qword ptr [r13 + r9*8]
    lea rdx, [rip + s_user]
    call str_eq
    lea r10, [rip + pl_user]
    test eax, eax
    jnz 4f
    mov rcx, qword ptr [r13 + r9*8]
    lea rdx, [rip + s_assistant]
    call str_eq
    lea r10, [rip + pl_asst]
    test eax, eax
    jnz 4f
    lea r10, [rip + pl_sys]             // any other role is labelled "System: "
4:  PUT_R(r10)
    mov r10, qword ptr [r14 + r9*8]
    PUT_R(r10)
    PUT_Z(cm_nl)
    inc r9
    jmp .Lcr_pl_turn
.Lcr_pl_end:
    PUT_Z(pl_end)
    jmp .Lcr_done

    // ---- Llama 3 ----
.Lcr_llama3:
    PUT_Z(l3_bos)
    test ebp, ebp
    jz 5f
    PUT_Z(l3_sys)
    mov rdx, qword ptr [r12 + TK_SYSPTR]
    mov r8, qword ptr [r12 + TK_SYSLEN] // the prompt has no NUL in the file buffer
    call put
    PUT_Z(l3_eot)
5:  xor r9d, r9d
6:  cmp r9, r15
    jge 7f
    PUT_Z(l3_hdr)
    mov r10, qword ptr [r13 + r9*8]
    PUT_R(r10)
    PUT_Z(l3_hdr_end)
    mov r10, qword ptr [r14 + r9*8]
    PUT_R(r10)
    PUT_Z(l3_eot)
    inc r9
    jmp 6b
7:  PUT_Z(l3_asst)
    jmp .Lcr_done

    // ---- ChatML (Qwen, SmolLM2) ----
.Lcr_chatml:
    test ebp, ebp
    jz 8f
    PUT_Z(cm_sys)
    mov rdx, qword ptr [r12 + TK_SYSPTR]
    mov r8, qword ptr [r12 + TK_SYSLEN]
    call put
    PUT_Z(cm_end)
8:  xor r9d, r9d
9:  cmp r9, r15
    jge 10f
    PUT_Z(cm_start)
    mov r10, qword ptr [r13 + r9*8]
    PUT_R(r10)
    PUT_Z(cm_nl)
    mov r10, qword ptr [r14 + r9*8]
    PUT_R(r10)
    PUT_Z(cm_end)
    inc r9
    jmp 9b
10: PUT_Z(cm_asst)

.Lcr_done:
    mov byte ptr [rbx + rsi], 0         // out[p] = '\0'
    mov rax, rsi
    add rsp, 40
    pop rbp
    pop r15
    pop r14
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret
