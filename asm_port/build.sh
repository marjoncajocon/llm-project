#!/usr/bin/env bash
#
# build.sh -- build the pure-assembly LLM engine (asm_port/).
#
#   ./build.sh            assemble every asm_port/*.S and link build/llm.exe
#   ./build.sh test NAME [extra.c ...]
#                         build tests/NAME.c + all asm objects (+ extra C files,
#                         e.g. ../quant.c) into build/NAME.exe and run it
#                         (ASM="tok tok_unitab" links only those asm modules)
#   ./build.sh clean
#
# Environment: ZIG=path to zig.exe (default /d/env/zig/zig.exe)
#
# The product is assembly only: -nostdlib, entry mainCRTStartup, kernel32 only.
# Tests may be C (they compare the asm against the repo's C reference code).
set -euo pipefail
cd "$(dirname "$0")"
ZIG=${ZIG:-/d/env/zig/zig.exe}
TARGET=x86_64-windows-gnu
B=build
mkdir -p "$B/obj"

asm_objs() {
  local objs=()
  for s in *.S; do
    [ -e "$s" ] || continue
    local o="$B/obj/${s%.S}.o"
    if [ ! -e "$o" ] || [ "$s" -nt "$o" ] || [ model.inc -nt "$o" ]; then
      "$ZIG" cc -target $TARGET -c "$s" -o "$o" >&2
    fi
    objs+=("$o")
  done
  echo "${objs[@]}"
}

case "${1:-all}" in
  all)
    OBJS=$(asm_objs)
    if [ ! -e main.S ]; then echo "objects built (no main.S yet, nothing to link)"; exit 0; fi
    "$ZIG" cc -target $TARGET -nostdlib -o "$B/llm.exe" $OBJS -lkernel32 -lshell32 \
      -Wl,--entry=mainCRTStartup -Wl,--subsystem,console -Wl,--stack,8388608
    echo "built $B/llm.exe"
    ;;
  test)
    name=$2; shift 2
    # tests link everything except main.o (the test has its own main), or only
    # the modules named in ASM="tok tok_unitab" (so one module's unfinished
    # file can't break another module's test)
    if [ -n "${ASM:-}" ]; then
      OBJS=""
      for m in $ASM; do
        o="$B/obj/$m.o"
        if [ ! -e "$o" ] || [ "$m.S" -nt "$o" ] || [ model.inc -nt "$o" ]; then
          "$ZIG" cc -target $TARGET -c "$m.S" -o "$o"
        fi
        OBJS="$OBJS $o"
      done
    else
      OBJS=$(asm_objs | tr ' ' '\n' | grep -v '/main\.o$' | tr '\n' ' ')
    fi
    "$ZIG" cc -target $TARGET -O2 -mavx2 -mfma -I.. -o "$B/$name.exe" "tests/$name.c" "$@" $OBJS -lkernel32
    "./$B/$name.exe"
    ;;
  clean) rm -rf "$B" ;;
  *) sed -n '3,15p' "$0"; exit 2 ;;
esac
