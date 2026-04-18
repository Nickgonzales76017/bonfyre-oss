CC          ?= cc
CFLAGS      ?= -O2 -Wall -Wextra -std=c11
LOCAL_CFLAGS = -I../../lib/libbonfyre/include
TARGET       = bonfyre-run
SRC          = src/main.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(LOCAL_CFLAGS) -o $(TARGET) $(SRC) -L../../lib/libbonfyre -lbonfyre -lsqlite3

clean:
	rm -f $(TARGET)

.PHONY: all clean
