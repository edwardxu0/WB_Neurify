CC=gcc
CFLAGS=-c -Wall
PROFILE_FLAGS=-fprofile-arcs -ftest-coverage
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
LPSOLVE_HOME=$(SRC_DIR)/lp_dev
LDFLAGS= -L$(OPENBLAS_HOME)/lib -lopenblas -lpthread -lm
LPFLAGS= $(LPSOLVE_HOME)/liblpsolve55.so -lm -ldl

all: gcov_report.html
	
check_tests: $(SRC_DIR)/interval.o $(SRC_DIR)/matrix.o $(SRC_DIR)/nnet.o $(SRC_DIR)/split.o check_neurify.o
	$(CC) $^ $(LDFLAGS) $(LPFLAGS) $(TST_LIBS) $(COV_LIBS) -o $@

test: check_tests
	./check_tests

gcov_report.html: test
	$(GCOV) $(GCOV_FLAGS)

%.o: %.c
	$(CC) -o $@ $(CFLAGS) $(PROFILE_FLAGS) $<

.PHONY: clean all

clean:
	rm -f *.o check_nnet_tests $(COV_FILES)
