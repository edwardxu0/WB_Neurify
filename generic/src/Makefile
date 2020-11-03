CC=gcc
OPENBLAS_HOME=/opt/OpenBLAS
CFLAGS=-c -Wall
LDFLAGS=-L$(OPENBLAS_HOME)/lib -lopenblas -lpthread -lm 
LPFLAGS=-llpsolve55 -lcolamd -ldl -lm 
INCLUDE_FLAGS=-I$(OPENBLAS_HOME)/include/

all: neurify
all: CFLAGS += -O3
all: LDFLAGS += -O3

debug: neurify
debug: CFLAGS += -DDEBUG -g -O0
debug: LDFLAGS += -DDEBUG -g -O0

neurify: matrix.o interval.o hpoly.o nnet.o split.o neurify.c
	$(CC) -o $@ $^ $(LDFLAGS) $(LPFLAGS) $(INCLUDE_FLAGS)

%.o: %.c
	$(CC) -o $@ $(INCLUDE_FLAGS) $(CFLAGS) $<

clean:
	rm -f *.o neurify

