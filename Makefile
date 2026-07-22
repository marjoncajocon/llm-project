# Makefile - Minimal GPT-2 Style Transformer
# All source files live flat in the project root (Lua-style layout).

CC      = gcc
CFLAGS  = -std=c11 -O3 -march=native -Wall -pthread
LDLIBS  = -lm -pthread

TARGET  = transformer
OBJS    = tensor.o matrix.o activation.o layernorm.o feedforward.o attention.o \
          embedding.o sampler.o model.o checkpoint.o tokenizer.o loss.o optim.o rmsnorm.o rope.o swiglu.o gqa.o quant.o llama.o \
          toolcall.o toolexec.o approval.o agent.o corpus.o train.o \
          image.o patch.o video.o audio.o imagegen.o audiogen.o selftest.o main.o

# --- Cross-compilation (uses `zig cc`; outputs to build/) ---
ZIG            ?= zig
BUILDDIR        = build
SRCS            = $(OBJS:.o=.c)
CROSS_TARGETS   = x86_64-linux-gnu aarch64-linux-gnu x86_64-linux-musl \
                  x86_64-windows-gnu aarch64-windows-gnu \
                  x86_64-macos aarch64-macos wasm32-wasi

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LDLIBS)

# Every .o is built from its matching .c in the root directory.
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Header dependencies (rebuild dependents when a header changes).
tensor.o:     tensor.h
matrix.o:     matrix.h tensor.h
activation.o: activation.h tensor.h
layernorm.o:  layernorm.h tensor.h
rmsnorm.o:    rmsnorm.h tensor.h
rope.o:       rope.h tensor.h
swiglu.o:     swiglu.h tensor.h matrix.h
gqa.o:        gqa.h tensor.h
quant.o:      quant.h
llama.o:      llama.h tensor.h matrix.h rmsnorm.h rope.h gqa.h swiglu.h quant.h
feedforward.o: feedforward.h tensor.h matrix.h activation.h
attention.o:  attention.h tensor.h matrix.h activation.h layernorm.h feedforward.h
embedding.o:  embedding.h tensor.h
sampler.o:    sampler.h tensor.h
model.o:      model.h tensor.h embedding.h attention.h layernorm.h matrix.h feedforward.h sampler.h
checkpoint.o: checkpoint.h model.h tensor.h
tokenizer.o:  tokenizer.h
loss.o:       loss.h tensor.h activation.h
optim.o:      optim.h tensor.h
toolcall.o:   toolcall.h
toolexec.o:   toolexec.h toolcall.h
approval.o:   approval.h
agent.o:      agent.h approval.h toolcall.h toolexec.h
corpus.o:     corpus.h
train.o:      train.h tensor.h tokenizer.h model.h loss.h optim.h checkpoint.h corpus.h
image.o:      image.h
patch.o:      patch.h tensor.h image.h matrix.h
video.o:      video.h tensor.h image.h patch.h
audio.o:      audio.h tensor.h matrix.h
imagegen.o:   imagegen.h image.h
audiogen.o:   audiogen.h audio.h
selftest.o:   selftest.h tensor.h loss.h optim.h sampler.h model.h checkpoint.h tokenizer.h image.h patch.h video.h audio.h imagegen.h audiogen.h rmsnorm.h rope.h swiglu.h gqa.h quant.h llama.h matrix.h
main.o:       selftest.h tensor.h loss.h optim.h sampler.h model.h checkpoint.h tokenizer.h train.h agent.h imagegen.h audiogen.h

# Cross-compile a binary for every target in CROSS_TARGETS into build/.
# Requires `zig` and a POSIX shell (Linux/macOS/WSL). No per-OS toolchain needed.
cross:
	@mkdir -p $(BUILDDIR)
	@for t in $(CROSS_TARGETS); do \
	  ext=""; lm="-lm"; \
	  case $$t in \
	    *windows*) ext=".exe" ;; \
	    wasm*)     ext=".wasm"; lm="" ;; \
	    *macos*)   lm="" ;; \
	  esac; \
	  out="$(BUILDDIR)/transformer-$$t$$ext"; \
	  echo "  [$$t] -> $$out"; \
	  $(ZIG) cc -target $$t -std=c11 -O2 -Wall -o "$$out" $(SRCS) $$lm || exit 1; \
	done
	@echo "cross builds written to $(BUILDDIR)/"

clean:
	rm -f $(OBJS) $(TARGET)
	rm -f $(BUILDDIR)/transformer-*

.PHONY: all clean cross
