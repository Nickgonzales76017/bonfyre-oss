CC      ?= cc
CFLAGS  ?= -O2 -Wall -Wextra -std=c11
LDFLAGS ?= -lsqlite3 -lpthread

all: bonfyre-queue

bonfyre-queue: src/main.c
	$(CC) $(CFLAGS) -o bonfyre-queue src/main.c $(LDFLAGS)

clean:
	rm -f bonfyre-queue
