CC      = cc
CFLAGS  = -std=c11 -O2 -Wall -Wextra -Wpedantic
LDFLAGS = -lsqlite3
TARGET  = bonfyre-flow
SRC     = src/main.c

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $(SRC) $(LDFLAGS)

clean:
	rm -f $(TARGET)
