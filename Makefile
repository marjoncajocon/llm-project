# Makefile - Minimal GPT-2 Style Transformer
# All source files live flat in the project root (Lua-style layout).

CC      = gcc
CFLAGS  = -std=c11 -O3 -march=native -Wall
LDLIBS  = -lm

TARGET  = transformer
OBJS    = tensor.o matrix.o activation.o layernorm.o feedforward.o attention.o \
          embedding.o sampler.o model.o checkpoint.o tokenizer.o loss.o optim.o main.o

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
feedforward.o: feedforward.h tensor.h matrix.h activation.h
attention.o:  attention.h tensor.h matrix.h activation.h layernorm.h feedforward.h
embedding.o:  embedding.h tensor.h
sampler.o:    sampler.h tensor.h
model.o:      model.h tensor.h embedding.h attention.h layernorm.h matrix.h feedforward.h sampler.h
checkpoint.o: checkpoint.h model.h tensor.h
tokenizer.o:  tokenizer.h
loss.o:       loss.h tensor.h activation.h
optim.o:      optim.h tensor.h
main.o:       tensor.h loss.h optim.h model.h checkpoint.h tokenizer.h

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
