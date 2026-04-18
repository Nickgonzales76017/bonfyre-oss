CC          ?= cc
CFLAGS      ?= -O2 -Wall -Wextra -std=c11
LOCAL_CFLAGS = -I../../lib/libbonfyre/include
LDFLAGS      = -L../../lib/libbonfyre -lbonfyre -lsqlite3 -lpthread
TARGET       = bonfyre-api

$(TARGET): src/main.c
	$(CC) $(CFLAGS) $(LOCAL_CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGET)
