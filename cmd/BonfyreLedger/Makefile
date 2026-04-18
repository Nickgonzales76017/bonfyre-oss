CC ?= cc
CFLAGS ?= -O2 -Wall -Wextra -std=c11
LOCAL_CFLAGS = -I../../lib/libbonfyre/include
LDFLAGS_BF = -L../../lib/libbonfyre -lbonfyre
TARGET = bonfyre-ledger
SRC = src/main.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(LOCAL_CFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS_BF)

clean:
	rm -f $(TARGET)

.PHONY: all clean
