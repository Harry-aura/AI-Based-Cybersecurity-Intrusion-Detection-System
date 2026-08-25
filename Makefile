CC = gcc
CFLAGS = -Wall -Wextra
TARGET = process_control

all: $(TARGET)

$(TARGET): src/process_control.c
	$(CC) $(CFLAGS) src/process_control.c -o $(TARGET)

clean:
	rm -f $(TARGET)
