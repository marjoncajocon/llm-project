# build/

Cross-compiled binaries land here. Everything in this folder is **git-ignored except
text/markdown files** (like this README) — the compiled binaries are build artifacts, not
source, so they're never committed.

## Generate them

```sh
make cross          # builds every target in the Makefile's CROSS_TARGETS into build/
```

`make cross` uses `zig cc` (bundles every toolchain — no per-OS setup) and a POSIX shell,
so run it on Linux/macOS/WSL. Clean them with `make clean`.

## What you get

| File | Platform |
| --- | --- |
| `transformer-x86_64-linux-gnu`     | Linux x86-64 (glibc) |
| `transformer-aarch64-linux-gnu`    | Linux ARM64 (glibc) |
| `transformer-x86_64-linux-musl`    | Linux x86-64 (musl, static-friendly) |
| `transformer-x86_64-windows-gnu.exe`  | Windows x86-64 |
| `transformer-aarch64-windows-gnu.exe` | Windows ARM64 |
| `transformer-x86_64-macos`         | macOS Intel |
| `transformer-aarch64-macos`        | macOS Apple Silicon |
| `transformer-wasm32-wasi.wasm`     | WebAssembly (WASI) |

Run the `.wasm` with any WASI runtime, e.g. `wasmtime --dir=. transformer-wasm32-wasi.wasm`.
See the top-level `README.md` ("Supported platforms") for details and caveats.
