CC=gcc
CFLAGS=-c -Wall
PROFILE_FLAGS=-fprofile-arcs -ftest-coverage -DDEBUG -g
TST_LIBS=-lcheck -lm -lpthread -lrt -lsubunit 
COV_LIBS=-lgcov -coverage
SRC_DIR=../src
TST_DIR=.
SRC_FILES=$(addprefix $(SRC_DIR)/, *.c) 
TST_FILES=$(addprefix $(TST_DIR)/, *.c)
COV_FILES=*.gcda *.gcno *.html
GCOV=gcovr 
GCOV_FLAGS=-r .. --html --html-details -o gcov_report.html

OPENBLAS_HOME=/opt/OpenBLAS
LDFLAGS=-L$(OPENBLAS_HOME)/lib -lopenblas -lpthread -lm
LPFLAGS=-llpsolve55 -lcolamd -ldl -lm 

all: gcov_report.html

%.src.o: $(SRC_DIR)/%.c
	$(CC) -o $@ $(CFLAGS) $(PROFILE_FLAGS) $<

check_neurify.o: check_neurify.c check_nnet.c check_split.c
	$(CC) -o $@ $(CFLAGS) $<

check_tests: interval.src.o matrix.src.o nnet.src.o split.src.o check_neurify.o
	$(CC) $^ $(LDFLAGS) $(LPFLAGS) $(TST_LIBS) $(COV_LIBS) $(PROFILE_FLAGS) -o $@

test: check_tests
	./check_tests

gcov_report.html: test
	$(GCOV) $(GCOV_FLAGS)

.PHONY: clean all

clean:
	rm -f *.o check_nnet_tests $(COV_FILES)
