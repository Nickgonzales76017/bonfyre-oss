CC ?= cc
CFLAGS ?= -O2 -Wall -Wextra -std=c11
LOCAL_CFLAGS = -I../../lib/libbonfyre/include
LDFLAGS_BF = -L../../lib/libbonfyre -lbonfyre

all: bonfyre-distribute

bonfyre-distribute: src/main.c
	$(CC) $(CFLAGS) $(LOCAL_CFLAGS) -o bonfyre-distribute src/main.c $(LDFLAGS_BF)

clean:
	rm -f bonfyre-distribute
