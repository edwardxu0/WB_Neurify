#ifndef _INTERVALH_
#define _INTERVALH_
#include "matrix.h"

struct Interval
{
	struct Matrix *lower_matrix;
	struct Matrix *upper_matrix;
};

struct Interval *Interval_new(int numRows, int numCols);
void destroy_Interval(struct Interval *interval);

#endif