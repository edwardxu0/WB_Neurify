#include "interval.h"

struct Interval *Interval_new(int numRows, int numCols)
{
    struct Interval *interval = malloc(sizeof(struct Interval));
    interval->lower_matrix = Matrix_new(numRows, numCols);
    interval->upper_matrix = Matrix_new(numRows, numCols);
    return interval;
}

void destroy_Interval(struct Interval *interval)
{
    destroy_Matrix(interval->lower_matrix);
    destroy_Matrix(interval->upper_matrix);
    free(interval);
}