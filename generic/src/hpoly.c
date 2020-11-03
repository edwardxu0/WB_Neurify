#include "hpoly.h"

struct HPoly *HPoly_new(int numVars, int numConstraints)
{
    struct HPoly *hpoly = malloc(sizeof(struct HPoly));
    hpoly->A = Matrix_new(numConstraints, numVars);
    hpoly->b = Matrix_new(numConstraints, 1);
    hpoly->num_vars = numVars;
    hpoly->num_constraints = numConstraints;
    return hpoly;
}

void destroy_HPoly(struct HPoly *hpoly)
{
    destroy_Matrix(hpoly->A);
    destroy_Matrix(hpoly->b);
    free(hpoly);
}