CC=gcc
OPENBLAS_HOME=/opt/OpenBLAS
LPSOLVE_HOME=lp_dev
CFLAGS=-c -Wall
LDFLAGS=-L$(OPENBLAS_HOME)/lib -lopenblas -lpthread -lm
LPFLAGS=$(LPSOLVE_HOME)/liblpsolve55.so -lm -ldl
INCLUDE_FLAGS=-I$(OPENBLAS_HOME)/include/

all: neurify
all: CFLAGS += -O3
all: LDFLAGS += -O3

debug: neurify
debug: CFLAGS += -DDEBUG -g
debug: LDFLAGS += -DDEBUG -g

neurify: matrix.o interval.o nnet.o split.o neurify.c
	$(CC) -o $@ $^ $(LDFLAGS) $(LPFLAGS) $(INCLUDE_FLAGS)

%.o: %.c
	$(CC) -o $@ $(INCLUDE_FLAGS) $(CFLAGS) $<

clean:
	rm -f *.o neurify

