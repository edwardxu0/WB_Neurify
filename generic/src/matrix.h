#ifndef _MATRIXH_
#define _MATRIXH_

#include <cblas.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

struct Matrix
{
    float *data;
    int row, col;
};

struct Matrix *Matrix_new(int numRows, int numCols);
void destroy_Matrix(struct Matrix *matrix);

void add_constant(struct Matrix *A, float alpha);

void matmul_with_factor(struct Matrix *A, struct Matrix *B, struct Matrix *C,
                        float alpha, float beta);

void matmul(struct Matrix *A, struct Matrix *B, struct Matrix *C);

void matmul_with_bias(struct Matrix *A, struct Matrix *B, struct Matrix *C);

void multiply(struct Matrix *A, struct Matrix *B);

void printMatrix(struct Matrix *A);

void fprintMatrix(FILE *fp, struct Matrix *A);

void relu(struct Matrix *A);

#endif