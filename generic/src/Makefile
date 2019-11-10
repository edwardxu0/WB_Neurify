CC=gcc
OPENBLAS_HOME=/opt/OpenBLAS
LPSOLVE_HOME=lp_dev
CFLAGS=-c -Wall
LDFLAGS=-I$(OPENBLAS_HOME)/include/ -L$(OPENBLAS_HOME)/lib -lopenblas -lpthread -lm
LPFLAGS=$(LPSOLVE_HOME)/liblpsolve55.so -lm -ldl

all: neurify
all: CFLAGS += -O3
all: LDFLAGS += -O3

debug: neurify
debug: CFLAGS += -DDEBUG -g
debug: LDFLAGS += -DDEBUG -g

neurify: matrix.o interval.o nnet.o split.o neurify.c
	$(CC) -o $@ $^ $(LDFLAGS) $(LPFLAGS)

%.o: %.c
	$(CC) -o $@ $(CFLAGS) $<

clean:
	rm -f *.o neurify

