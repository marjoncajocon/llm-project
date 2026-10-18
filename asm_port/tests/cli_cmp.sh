#!/usr/bin/env bash
# cli_cmp.sh - byte-compare `transformer hfgen|hfchat` (C reference build) with
# `llm.exe hfgen|hfchat`: stdout, stderr and exit code must all be identical.
#
#   R=path/to/transformer.exe L=asm_port/build/llm.exe W=/some/tmp/dir \
#     bash asm_port/tests/cli_cmp.sh [filter]
#
# Needs model/qwen/model_int8.bin, model/smollm/model_int8.bin (+ tok.hf) and,
# optionally, the release folder E:/llm-release (agent models, used read-only).
# SYN = folder with the synthetic sliding-window checkpoint syn_cli_q.bin
# (windows 6/0, max_seq 40); copies with window 8 and 16 are made in $W so the
# --samples rewind is tested on row-major AND 8-slot-blocked K rings.
set -u
cd "$(dirname "$0")/../.."
R=${R:?set R to the reference transformer.exe}
L=${L:-asm_port/build/llm.exe}
W=${W:?set W to a scratch directory}
SYN=${SYN:-}
REL=${REL:-/e/llm-release}
F=${1:-}
mkdir -p "$W/in"
python asm_port/tests/cli_inputs.py "$W/in" >/dev/null
pass=0; fail=0
t() {   # t NAME INPUT-FILE ARGS...
  local name="$1" in="$2"; shift 2
  [ -n "$F" ] && [[ "$name" != *$F* ]] && return
  "$R" "$@" <"$in" >"$W/c.out" 2>"$W/c.err"; echo "rc=$?" >>"$W/c.out"
  local t0=$(date +%s%N)
  "$L" "$@" <"$in" >"$W/a.out" 2>"$W/a.err"; echo "rc=$?" >>"$W/a.out"
  local ms=$(( ($(date +%s%N) - t0) / 1000000 ))
  if cmp -s "$W/c.out" "$W/a.out" && cmp -s "$W/c.err" "$W/a.err"; then
    pass=$((pass+1)); echo "SAME  $name ($(wc -c <"$W/c.out") B out, $(wc -c <"$W/c.err") B err, asm ${ms} ms)"
  else
    fail=$((fail+1)); echo "DIFF  $name"
    diff <(od -c "$W/c.out") <(od -c "$W/a.out") | head -8
    diff "$W/c.err" "$W/a.err" | head -6
  fi
}
I=$W/in
QM=model/qwen/model_int8.bin; QT=model/qwen/tok.hf
SM=model/smollm/model_int8.bin; ST=model/smollm/tok.hf

# ---- usage / argument errors ----
t usage-hfgen        $I/gen_france.in hfgen
t usage-hfgen-1      $I/gen_france.in hfgen $SM
t usage-samples0     $I/gen_france.in hfgen $SM $ST 5 0 --samples 0
t usage-samples65    $I/gen_france.in hfgen $SM $ST 5 0 --samples 65
t usage-samples-last $I/gen_france.in hfgen $SM $ST 5 0 --samples
t usage-samples-abc  $I/gen_france.in hfgen $SM $ST 5 0 --samples abc
t usage-bogus        $I/gen_france.in hfgen $SM $ST 5 0 --bogus
t usage-hfchat       $I/chat_single.in hfchat
t usage-hfchat-1     $I/chat_single.in hfchat $QM
t tok-missing        $I/gen_france.in hfgen $SM nope.hf 5 0
t tok-notok          $I/gen_france.in hfgen $SM $SM 5 0
t chat-tok-missing   $I/chat_single.in hfchat $QM nope.hf 5
t chat-model-missing $I/chat_single.in hfchat nope.bin $QT 5

# ---- hfgen subcommand, --stop-tags ----
t gen-sub            $I/gen_france.in hfgen $SM $ST 24 0
t gen-sub-qwen       $I/gen_qwen_chat.in hfgen $QM $QT 40 0.7 5
t stop-tags-tool     $I/gen_tool_text.in hfgen $SM $ST 60 0 --stop-tags
t stop-tags-first    $I/gen_tool_text.in hfgen --stop-tags $SM $ST 60 0.9 3
t stop-tags-qwen     $I/gen_qwen_chat.in hfgen $QM $QT 30 0 --stop-tags
if [ -e "$REL/models/sft/model.bin" ]; then
  for p in gen_open gen_save; do
    t rel-sft-$p     $I/$p.in hfgen "$REL/models/sft/model.bin" "$REL/tok.hf" 160 0.1 --stop-tags
    t rel-base-$p    $I/$p.in hfgen "$REL/models/base/model.bin" "$REL/tok.hf" 160 0.1 --stop-tags
  done
fi

# ---- --samples ----
for n in 1 3 8; do
  t samples$n-qwen   $I/gen_qwen_chat.in hfgen $QM $QT 24 0.8 7 --samples $n
  t samples$n-smol   $I/gen_france.in hfgen $SM $ST 24 0.9 11 --samples $n
done
t samples3-greedy    $I/gen_france.in hfgen $SM $ST 12 0 --samples 3
t samples3-stoptags  $I/gen_tool_text.in hfgen $SM $ST 40 1.0 2 --samples 3 --stop-tags
t samples-seedwrap   $I/gen_france.in hfgen $SM $ST 8 1.0 18446744073709551614 --samples 4
t samples64          $I/gen_france.in hfgen $SM $ST 3 1.0 1 --samples 64
t samples-n0         $I/gen_france.in hfgen $SM $ST 0 1.0 1 --samples 3
t samples-empty      $I/chat_empty.in hfgen $SM $ST 5 1.0 1 --samples 3
if [ -n "$SYN" ] && [ -e "$SYN/syn_cli_q.bin" ]; then
  # ring layers: window 6 (row-major K), 8 and 16 (blocked K); max_seq 40
  for wnd in 8 16; do
    python - "$SYN/syn_cli_q.bin" "$W/syn_w$wnd.bin" $wnd <<'PY'
import struct, sys
b = bytearray(open(sys.argv[1], "rb").read())
assert struct.unpack_from("<I", b, 48)[0] & 2          # windows present
struct.pack_into("<i", b, 52, int(sys.argv[3]))         # layer 0 window
open(sys.argv[2], "wb").write(b)
PY
  done
  for m in "$SYN/syn_cli_q.bin" "$W/syn_w8.bin" "$W/syn_w16.bin" "$SYN/syn_cli_f.bin"; do
    b=$(basename "$m" .bin)
    t ring-$b-s5      $I/gen_france.in hfgen "$m" $ST 30 1.0 5 --samples 5
    t ring-$b-long    $I/chat_multi.in hfgen "$m" $ST 25 1.0 9 --samples 3
  done
fi

# ---- hfchat ----
for c in chat_single chat_system chat_multi chat_multi_sys chat_unicode chat_empty chat_nosep \
         chat_role_only chat_odd_roles chat_crlf chat_nul chat_ctrlz; do
  t qwen-$c          $I/$c.in hfchat $QM $QT 48
done
t qwen-chat_code-512 $I/chat_code.in hfchat $QM $QT 512
t qwen-default-n     $I/chat_single.in hfchat $QM $QT
t qwen-n0            $I/chat_single.in hfchat $QM $QT 0
t qwen-extra-args    $I/chat_single.in hfchat $QM $QT 20 0.1 --whatever
t smol-chat_multi    $I/chat_multi.in hfchat $SM $ST 40
t smol-chat_system   $I/chat_system.in hfchat $SM $ST 40
t smol-chat_big      $I/chat_big.in hfchat $SM $ST 16
if [ -e "$REL/models/sft/model.bin" ]; then
  t rel-tok-chat     $I/chat_multi.in hfchat $SM "$REL/tok.hf" 30
fi
echo "== $pass same, $fail different"
[ $fail -eq 0 ]
