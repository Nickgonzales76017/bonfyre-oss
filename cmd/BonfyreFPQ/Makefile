CC ?= cc
CFLAGS ?= -O2 -Wall -Wextra -std=c11
LOCAL_CFLAGS = -Iinclude
LDFLAGS = -lm

TARGET = bonfyre-fpq
BUILD = build

SRC = src/fwht.c \
	src/polar.c \
	src/seed.c \
	src/qjl.c \
	src/debruijn.c \
	src/fpq_codec.c \
	src/ggml_reader.c \
	src/serialize.c \
	src/v4_optimizations.c \
	src/main.c

OBJ = $(SRC:src/%.c=$(BUILD)/%.o)

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) $(LOCAL_CFLAGS) -o $(TARGET) $(OBJ) $(LDFLAGS)

$(BUILD)/%.o: src/%.c include/fpq.h | $(BUILD)
	$(CC) $(CFLAGS) $(LOCAL_CFLAGS) -c $< -o $@

$(BUILD):
	mkdir -p $(BUILD)

clean:
	rm -rf $(BUILD) $(TARGET)

.PHONY: all clean
