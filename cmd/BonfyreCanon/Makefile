CC       ?= cc
CFLAGS   ?= -O2 -Wall -Wextra -std=c11
LOCAL_CFLAGS = -I/opt/homebrew/include -Isrc -I../../lib/libbonfyre/include
LDFLAGS  = -L../../lib/libbonfyre -lbonfyre -L/opt/homebrew/lib -ltree-sitter
TARGET    = bonfyre-canon
SRC       = src/main.c src/ts_json_parser.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(LOCAL_CFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

clean:
	rm -f $(TARGET)
