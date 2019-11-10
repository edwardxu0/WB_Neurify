#ifndef _NEURIFYCHECKH_
#define _NEURIFYCHECKH_

void assert_float_close(float x, float y, float tol)
{
    if (abs(x - y) > tol)
        ck_assert_msg(0,
                      "Assertion 'abs(x - y) > %f' failed: x == %f, y == %f",
                      tol, x, y);
}

#endif