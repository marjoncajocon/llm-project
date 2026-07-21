# Makefile - Minimal GPT-2 Style Transformer
# All source files live flat in the project root (Lua-style layout).

CC      = gcc
CFLAGS  = -std=c11 -O3 -march=native -Wall
LDLIBS  = -lm

TARGET  = transformer
OBJS    = tensor.o matrix.o activation.o layernorm.o feedforward.o attention.o main.o

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
main.o:       tensor.h attention.h

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
