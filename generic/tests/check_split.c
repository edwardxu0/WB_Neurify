#include <check.h>
#include <stdlib.h>
#include "check.h"
#include "../src/split.h"

START_TEST(test_check)
{
    struct NNet *nnet = load_conv_network("artifacts/example1.nnet");
    struct Interval *output_constraint = Interval_new(nnet->outputSize, 1);
    for (int i = 0; i < nnet->outputSize; i++)
    {
        output_constraint->lower_matrix->data[i] = 0;
        output_constraint->upper_matrix->data[i] = 1;
    }
    nnet->output_constraint = output_constraint;

    // within bounds
    struct Interval *output_interval = Interval_new(nnet->outputSize, 1);
    for (int i = 0; i < nnet->outputSize; i++)
    {
        output_interval->lower_matrix->data[i] = 0;
        output_interval->upper_matrix->data[i] = 1;
    }
    ck_assert_int_eq(check(nnet, output_interval), 0);
    for (int i = 0; i < nnet->outputSize; i++)
    {
        output_interval->lower_matrix->data[i] = 0.1;
        output_interval->upper_matrix->data[i] = 0.9;
    }
    ck_assert_int_eq(check(nnet, output_interval), 0);

    // violate lower bound
    for (int i = 0; i < nnet->outputSize; i++)
    {
        output_interval->lower_matrix->data[i] = -0.1;
        output_interval->upper_matrix->data[i] = 0.9;
    }
    ck_assert_int_eq(check(nnet, output_interval), 1);
    for (int i = 0; i < nnet->outputSize; i++)
    {
        output_interval->lower_matrix->data[i] = -0.5;
        output_interval->upper_matrix->data[i] = -0.2;
    }
    ck_assert_int_eq(check(nnet, output_interval), 1);

    // violate upper bound
    for (int i = 0; i < nnet->outputSize; i++)
    {
        output_interval->lower_matrix->data[i] = 0.3;
        output_interval->upper_matrix->data[i] = 1.5;
    }
    ck_assert_int_eq(check(nnet, output_interval), 1);
    for (int i = 0; i < nnet->outputSize; i++)
    {
        output_interval->lower_matrix->data[i] = 1.1;
        output_interval->upper_matrix->data[i] = 1.9;
    }
    ck_assert_int_eq(check(nnet, output_interval), 1);

    destroy_conv_network(nnet);
    destroy_Interval(output_constraint);
    destroy_Interval(output_interval);
}
END_TEST

Suite *split_suite(void)
{
    Suite *s = suite_create("split");

    TCase *tc_check = tcase_create("check");
    tcase_add_test(tc_check, test_check);
    suite_add_tcase(s, tc_check);

    return s;
}
