#ifndef _HPOLYH_
#define _HPOLYH_
#include "matrix.h"

struct HPoly
{
	struct Matrix *A;
	struct Matrix *b;
	int num_vars;
	int num_constraints;
};

struct HPoly *HPoly_new(int numVars, int numConstraints);
void destroy_HPoly(struct HPoly *hpoly);

#endif