#!/usr/bin/env mmc
# build.sh - build the engine from the flat C sources with zig cc (any target).
#
# Run it with the mmc shell (D:\mmc-shell) -- or any bash:
#   D:\mmc-shell\mmc.exe build.sh                native Windows -> transformer.exe
#   D:\mmc-shell\mmc.exe build.sh test           native build, then run the selftest
#   D:\mmc-shell\mmc.exe build.sh debug          native -O0 -g build
#   D:\mmc-shell\mmc.exe build.sh linux          cross -> build/transformer-x86_64-linux
#   D:\mmc-shell\mmc.exe build.sh linux-arm      cross -> build/transformer-aarch64-linux
#   D:\mmc-shell\mmc.exe build.sh macos          cross -> build/transformer-aarch64-macos
#   D:\mmc-shell\mmc.exe build.sh wasm           cross -> build/transformer.wasm
#   D:\mmc-shell\mmc.exe build.sh all            every cross target above
#   ZIG=... / CC=...                             override the compiler
#
# zig bundles every target's libc and linker, so no other toolchain is needed.
# Native builds use -march=native; cross builds use the target's baseline CPU.

set -e

ZIG="${ZIG:-D:/env/zig/zig.exe}"
MODE=release
RUN_TEST=0
TARGETS=""
for arg in "$@"; do
  case "$arg" in
    test)  RUN_TEST=1 ;;
    debug) MODE=debug ;;
    linux|linux-arm|macos|wasm) TARGETS="$TARGETS $arg" ;;
    all)   TARGETS="linux linux-arm macos wasm" ;;
    *)     echo "build.sh: unknown argument '$arg'"
           echo "  use: test debug linux linux-arm macos wasm all"; exit 2 ;;
  esac
done

if [ -n "$CC" ]; then
  COMPILER="$CC"
elif [ -x "$ZIG" ]; then
  COMPILER="$ZIG cc"
else
  COMPILER=gcc
fi

if [ "$MODE" = debug ]; then
  OPT="-O0 -g"
else
  OPT="-O3"
fi

# --- native (Windows) ---
if [ -z "$TARGETS" ]; then
  echo "compiler: $COMPILER"
  $COMPILER -std=c11 $OPT -march=native -Wall -pthread -o transformer.exe *.c -lm
  echo "built:    transformer.exe"
fi

# --- cross targets (zig only) ---
for t in $TARGETS; do
  case "$t" in
    linux)     TRIPLE=x86_64-linux-gnu;  OUT=build/transformer-x86_64-linux;  LIBS="-lm -pthread" ;;
    linux-arm) TRIPLE=aarch64-linux-gnu; OUT=build/transformer-aarch64-linux; LIBS="-lm -pthread" ;;
    macos)     TRIPLE=aarch64-macos;     OUT=build/transformer-aarch64-macos; LIBS="-pthread" ;;
    wasm)      TRIPLE=wasm32-wasi;       OUT=build/transformer.wasm;          LIBS="" ;;
  esac
  echo "cross:    $t ($TRIPLE)"
  "$ZIG" cc -target $TRIPLE -std=c11 $OPT -o "$OUT" *.c $LIBS
  echo "built:    $OUT"
done

if [ "$RUN_TEST" = 1 ]; then
  ./transformer.exe > build/selftest.log 2>&1 || true
  if grep -q "Output validation: PASSED" build/selftest.log; then
    echo "selftest: PASSED (log: build/selftest.log)"
  else
    grep -n "FAIL" build/selftest.log | head -20
    echo "selftest: FAILED (log: build/selftest.log)"
    exit 1
  fi
fi
