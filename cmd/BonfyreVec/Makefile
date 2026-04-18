CC          ?= cc
CFLAGS      ?= -O2 -Wall -Wextra -std=c11
LOCAL_CFLAGS = -I/opt/homebrew/opt/sqlite/include -I../../lib/libbonfyre/include
LDFLAGS      = -L/opt/homebrew/opt/sqlite/lib -lsqlite3 -L../../lib/libbonfyre -lbonfyre
TARGET       = bonfyre-vec
SRC          = src/main.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(LOCAL_CFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

clean:
	rm -f $(TARGET)
