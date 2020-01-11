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

START_TEST(test_check1)
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
    struct Matrix *output = Matrix_new(nnet->outputSize, 1);
    for (int i = 0; i < nnet->outputSize; i++)
    {
        output->data[i] = 0;
    }
    ck_assert_int_eq(check1(nnet, output), 0);
    for (int i = 0; i < nnet->outputSize; i++)
    {
        output->data[i] = 0.1;
    }
    ck_assert_int_eq(check1(nnet, output), 0);

    // violate lower bound
    for (int i = 0; i < nnet->outputSize; i++)
    {
        output->data[i] = -0.5;
    }
    ck_assert_int_eq(check1(nnet, output), 1);

    // violate upper bound
    for (int i = 0; i < nnet->outputSize; i++)
    {
        output->data[i] = 1.3;
    }
    ck_assert_int_eq(check1(nnet, output), 1);

    destroy_conv_network(nnet);
    destroy_Interval(output_constraint);
    destroy_Matrix(output);
}
END_TEST

START_TEST(test_check_adv1)
{
    struct NNet *nnet = load_conv_network("artifacts/example1.nnet");
    struct Interval *output_constraint = Interval_new(nnet->outputSize, 1);
    for (int i = 0; i < nnet->outputSize; i++)
    {
        output_constraint->lower_matrix->data[i] = 0;
        output_constraint->upper_matrix->data[i] = 1;
    }
    nnet->output_constraint = output_constraint;

    struct Matrix *possible_adv = Matrix_new(1, nnet->inputSize);
    for (int i = 0; i < nnet->inputSize; i++)
    {
        possible_adv->data[i] = 0;
    }
    adv_found = 0;
    check_adv1(nnet, possible_adv);
    ck_assert_int_eq(adv_found, 0);

    for (int i = 0; i < nnet->inputSize; i++)
    {
        possible_adv->data[i] = 1;
    }
    adv_found = 0;
    check_adv1(nnet, possible_adv);
    ck_assert_int_eq(adv_found, 1);

    destroy_conv_network(nnet);
    destroy_Interval(output_constraint);
    destroy_Matrix(possible_adv);
}
END_TEST

START_TEST(test_forward_prop_interval_equation_conv_lp_example1)
{
    struct NNet *nnet = load_conv_network("artifacts/example1.nnet");
    int inputSize = nnet->inputSize;
    int outputSize = nnet->outputSize;

    struct Interval *input_interval = Interval_new(1, inputSize);
    for (int i = 0; i < inputSize; i++)
    {
        input_interval->lower_matrix->data[i] = -1;
        input_interval->upper_matrix->data[i] = 1;
    }
    struct Interval *output_constraint = Interval_new(outputSize, 1);
    for (int i = 0; i < outputSize; i++)
    {
        output_constraint->lower_matrix->data[i] = 0;
        output_constraint->upper_matrix->data[i] = 1;
    }
    nnet->output_constraint = output_constraint;

    int num_nodes = 0;
    for (int layer = 1; layer < nnet->numLayers; layer++)
    {
        num_nodes += nnet->layerSizes[layer];
    }
    ERR_NODE = num_nodes;

    int sigs[num_nodes];
    memset(sigs, 0, sizeof(int) * num_nodes);
    int target = 0;
    int sig = 0;

    lprec *lp = make_lp(0, inputSize);
    set_verbose(lp, CRITICAL);
    set_input_constraints(input_interval, lp, inputSize);

    int need_to_split = forward_prop_interval_equation_conv_lp(nnet,
                                                               input_interval,
                                                               sigs,
                                                               target,
                                                               sig,
                                                               lp);
    ck_assert_int_eq(need_to_split, 1);

    sig = 1;
    need_to_split = forward_prop_interval_equation_conv_lp(nnet,
                                                           input_interval,
                                                           sigs,
                                                           target,
                                                           sig,
                                                           lp);
    ck_assert_int_eq(need_to_split, 0);

    delete_lp(lp);
    destroy_conv_network(nnet);
    destroy_Interval(input_interval);
    destroy_Interval(output_constraint);
}
END_TEST

START_TEST(test_forward_prop_interval_equation_conv_lp_example2)
{
    struct NNet *nnet = load_conv_network("artifacts/example2.nnet");
    int inputSize = nnet->inputSize;
    int outputSize = nnet->outputSize;

    struct Interval *input_interval = Interval_new(1, inputSize);
    for (int i = 0; i < inputSize; i++)
    {
        input_interval->lower_matrix->data[i] = -1;
        input_interval->upper_matrix->data[i] = 1;
    }
    struct Interval *output_constraint = Interval_new(outputSize, 1);
    for (int i = 0; i < outputSize; i++)
    {
        output_constraint->lower_matrix->data[i] = 0;
        output_constraint->upper_matrix->data[i] = 1;
    }
    nnet->output_constraint = output_constraint;

    int num_nodes = 0;
    for (int layer = 1; layer < nnet->numLayers; layer++)
    {
        num_nodes += nnet->layerSizes[layer];
    }
    ERR_NODE = num_nodes;

    int sigs[num_nodes];
    memset(sigs, 0, sizeof(int) * num_nodes);
    int target = 0;
    int sig = 0;

    lprec *lp = make_lp(0, inputSize);
    set_verbose(lp, CRITICAL);
    set_input_constraints(input_interval, lp, inputSize);

    int need_to_split = forward_prop_interval_equation_conv_lp(nnet,
                                                               input_interval,
                                                               sigs,
                                                               target,
                                                               sig,
                                                               lp);
    ck_assert_int_eq(need_to_split, 1);

    sig = 1;
    need_to_split = forward_prop_interval_equation_conv_lp(nnet,
                                                           input_interval,
                                                           sigs,
                                                           target,
                                                           sig,
                                                           lp);
    ck_assert_int_eq(need_to_split, 1);

    delete_lp(lp);
    destroy_conv_network(nnet);
    destroy_Interval(input_interval);
    destroy_Interval(output_constraint);
}
END_TEST

Suite *split_suite(void)
{
    Suite *s = suite_create("split");

    TCase *tc_check = tcase_create("check");
    tcase_add_test(tc_check, test_check);
    tcase_add_test(tc_check, test_check1);
    tcase_add_test(tc_check, test_check_adv1);
    suite_add_tcase(s, tc_check);

    TCase *tc_forward_prop_lp = tcase_create("forward_prop_lp");
    tcase_add_test(tc_forward_prop_lp,
                   test_forward_prop_interval_equation_conv_lp_example1);
    tcase_add_test(tc_forward_prop_lp,
                   test_forward_prop_interval_equation_conv_lp_example2);
    suite_add_tcase(s, tc_forward_prop_lp);

    return s;
}
