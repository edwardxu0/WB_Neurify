#include <check.h>
#include <stdlib.h>
#include "check.h"
#include "../src/nnet.h"

START_TEST(test_load_conv_network_invalid_filename)
{
    struct NNet *nnet = load_conv_network("<INVALID>");
    destroy_conv_network(nnet);
}
END_TEST

START_TEST(test_load_conv_network_example1)
{
    struct NNet *nnet = load_conv_network("artifacts/example1.nnet");

    ck_assert_int_eq(nnet->numLayers, 3);
    ck_assert_int_eq(nnet->inputSize, 2);
    ck_assert_int_eq(nnet->outputSize, 2);
    ck_assert_int_eq(nnet->maxLayerSize, 2);
    for (int i = 0; i < nnet->numLayers; i++)
    {
        ck_assert_int_eq(nnet->layerSizes[i], 2);
        ck_assert_int_eq(nnet->layerTypes[i], 0);
    }
    ck_assert_int_eq(nnet->convLayersNum, 0);

    destroy_conv_network(nnet);
}
END_TEST

START_TEST(test_load_conv_network_example2)
{
    struct NNet *nnet = load_conv_network("artifacts/example2.nnet");

    ck_assert_int_eq(nnet->numLayers, 5);
    ck_assert_int_eq(nnet->inputSize, 16);
    ck_assert_int_eq(nnet->outputSize, 1);
    ck_assert_int_eq(nnet->maxLayerSize, 16);

    ck_assert_int_eq(nnet->convLayersNum, 2);
    for (int i = 0; i < nnet->numLayers; i++)
    {
        if (i < nnet->convLayersNum)
        {
            ck_assert_int_eq(nnet->layerTypes[i], 1);
        }
        else
        {
            ck_assert_int_eq(nnet->layerTypes[i], 0);
        }
    }

    destroy_conv_network(nnet);
}
END_TEST

START_TEST(test_evaluate_conv_example1)
{
    struct NNet *nnet = load_conv_network("artifacts/example1.nnet");

    struct Matrix *input = Matrix_new(1, nnet->inputSize);
    struct Matrix *output = Matrix_new(nnet->outputSize, 1);
    float expected_output[nnet->outputSize];

    input->data[0] = 1;
    input->data[1] = 1;
    evaluate_conv(nnet, input, output);
    expected_output[0] = 5.0;
    expected_output[1] = 2.0;
    for (int i = 0; i < nnet->outputSize; i++)
    {
        assert_float_close(output->data[i], expected_output[i], 1e-6);
    }

    input->data[0] = -1;
    input->data[1] = 1;
    evaluate_conv(nnet, input, output);
    expected_output[0] = 1.0;
    expected_output[1] = 0.0;
    for (int i = 0; i < nnet->outputSize; i++)
    {
        assert_float_close(output->data[i], expected_output[i], 1e-6);
    }

    input->data[0] = 1;
    input->data[1] = -1;
    evaluate_conv(nnet, input, output);
    expected_output[0] = 3.0;
    expected_output[1] = 0.0;
    for (int i = 0; i < nnet->outputSize; i++)
    {
        assert_float_close(output->data[i], expected_output[i], 1e-6);
    }

    destroy_conv_network(nnet);
    destroy_Matrix(input);
    destroy_Matrix(output);
}
END_TEST

START_TEST(test_evaluate_conv_example2)
{
    struct NNet *nnet = load_conv_network("artifacts/example2.nnet");

    struct Matrix *input = Matrix_new(1, nnet->inputSize);
    struct Matrix *output = Matrix_new(nnet->outputSize, 1);
    float expected_output[nnet->outputSize];

    for (int i = 0; i < nnet->inputSize; i++)
    {
        input->data[i] = 1;
    }
    evaluate_conv(nnet, input, output);
    expected_output[0] = 2304.0;
    for (int i = 0; i < nnet->outputSize; i++)
    {
        assert_float_close(output->data[i], expected_output[i], 1e-6);
    }

    for (int i = 0; i < nnet->inputSize; i++)
    {
        input->data[i] = -1;
    }
    evaluate_conv(nnet, input, output);
    expected_output[0] = 0.0;
    for (int i = 0; i < nnet->outputSize; i++)
    {
        assert_float_close(output->data[i], expected_output[i], 1e-6);
    }

    for (int i = 0; i < nnet->inputSize; i++)
    {
        input->data[i] = i;
    }
    evaluate_conv(nnet, input, output);
    expected_output[0] = 17280.0;
    for (int i = 0; i < nnet->outputSize; i++)
    {
        assert_float_close(output->data[i], expected_output[i], 1e-6);
    }

    for (int i = 0; i < nnet->inputSize; i++)
    {
        input->data[i] = i - nnet->inputSize / 2;
    }
    evaluate_conv(nnet, input, output);
    expected_output[0] = 1728.0;
    for (int i = 0; i < nnet->outputSize; i++)
    {
        assert_float_close(output->data[i], expected_output[i], 1e-6);
    }

    destroy_conv_network(nnet);
    destroy_Matrix(input);
    destroy_Matrix(output);
}
END_TEST

START_TEST(test_sort)
{
    {
        float array[5] = {0.1, 0.2, 0.3, 0.4, 0.5};
        int ind[5] = {0, 1, 2, 3, 4};
        int length = 5;
        sort(array, length, ind);
        for (int i = 0; i < length; i++)
        {
            ck_assert_int_eq(ind[i], length - i - 1);
        }
    }

    {
        float array[5] = {0.1, 0.2, 0.3, 0.4, 0.5};
        int ind[5] = {4, 3, 2, 1, 0};
        int length = 5;
        sort(array, length, ind);
        for (int i = 0; i < length; i++)
        {
            ck_assert_int_eq(ind[i], length - i - 1);
        }
    }

    {
        float array[5] = {0.5, 0.2, 0.3, 0.1, 0.4};
        int ind[5] = {0, 1, 2, 3, 4};
        int sort_ind[5] = {0, 4, 2, 1, 3};
        int length = 5;
        sort(array, length, ind);
        for (int i = 0; i < length; i++)
        {
            ck_assert_int_eq(ind[i], sort_ind[i]);
        }
    }

    {
        float array[5] = {0.5, 0.2, 0.3, 0.1, 0.4};
        int ind[5] = {2, 3, 1, 0, 4};
        int sort_ind[5] = {0, 4, 2, 1, 3};
        int length = 5;
        sort(array, length, ind);
        for (int i = 0; i < length; i++)
        {
            ck_assert_int_eq(ind[i], sort_ind[i]);
        }
    }

    {
        float array[5] = {0.5, 0.5, 0.3, 0.4, 0.4};
        int ind[5] = {0, 1, 2, 3, 4};
        int sort_ind[5] = {0, 1, 3, 4, 2};
        int length = 5;
        sort(array, length, ind);
        for (int i = 0; i < length; i++)
        {
            ck_assert_int_eq(ind[i], sort_ind[i]);
        }
    }

    {
        float array[5] = {0.5, 0.5, 0.3, 0.4, 0.4};
        int ind[5] = {0, 1, 2, 3, 4};
        int length = 3;
        int sort_ind[3] = {0, 1, 2};
        sort(array, length, ind);
        for (int i = 0; i < length; i++)
        {
            ck_assert_int_eq(ind[i], sort_ind[i]);
        }
    }
}
END_TEST

START_TEST(test_sort_layers)
{
    {
        int numLayers = 5;
        int layerSizes[5] = {2, 4, 3, 2, 1}; // num_nodes = 4+3+2+1 = 10
        int wrong_node_length = 5;
        int wrong_nodes[5] = {0, 1, 3, 7, 9};
        int sorted_wrong_nodes[5] = {0, 1, 3, 7, 9};
        sort_layers(numLayers, layerSizes, wrong_node_length, wrong_nodes);
        for (int i = 0; i < wrong_node_length; i++)
        {
            ck_assert_int_eq(wrong_nodes[i], sorted_wrong_nodes[i]);
        }
    }

    {
        int numLayers = 5;
        int layerSizes[5] = {2, 4, 3, 2, 1};
        int wrong_node_length = 5;
        int wrong_nodes[5] = {0, 6, 3, 5, 9};
        int sorted_wrong_nodes[5] = {0, 3, 6, 5, 9};
        sort_layers(numLayers, layerSizes, wrong_node_length, wrong_nodes);
        for (int i = 0; i < wrong_node_length; i++)
        {
            ck_assert_int_eq(wrong_nodes[i], sorted_wrong_nodes[i]);
        }
    }

    {
        int numLayers = 5;
        int layerSizes[5] = {2, 4, 3, 2, 1};
        int wrong_node_length = 5;
        int wrong_nodes[5] = {9, 8, 5, 4, 2};
        int sorted_wrong_nodes[5] = {2, 5, 4, 8, 9};
        sort_layers(numLayers, layerSizes, wrong_node_length, wrong_nodes);
        for (int i = 0; i < wrong_node_length; i++)
        {
            ck_assert_int_eq(wrong_nodes[i], sorted_wrong_nodes[i]);
        }
    }
}
END_TEST

START_TEST(test_load_inputs)
{
    {
        int inputSize = 2;
        float input[inputSize];
        load_inputs("artifacts/test_inputs/example1.center", inputSize, input);
        for (int i = 0; i < inputSize; i++)
        {
            ck_assert_int_eq(input[i], 0.0);
        }
    }
    {
        int inputSize = 16;
        float input[inputSize];
        load_inputs("artifacts/test_inputs/example2.center", inputSize, input);
        for (int i = 0; i < inputSize; i++)
        {
            ck_assert_int_eq(input[i], 0.0);
        }
    }
}
END_TEST

START_TEST(test_load_inputs_not_exist)
{
    {
        int inputSize = 2;
        float input[inputSize];
        load_inputs("<INVALID>", inputSize, input);
    }
}
END_TEST

START_TEST(test_initialize_input_interval)
{
    int inputSize = 5;
    int epsilon[5] = {0.1,
                      0.02,
                      0.5,
                      1.2,
                      0.001};
    struct Interval *interval = Interval_new(1, inputSize);

    for (int i = 0; i < 5; i++)
    {
        float input[5] = {0.2, 0.3, 0.4, 0.5, 0.1};
        initialize_input_interval(interval, inputSize, input, epsilon[i]);
        for (int j = 0; j < inputSize; j++)
        {
            assert_float_close(interval->lower_matrix->data[j],
                               input[j] - epsilon[i],
                               1e-6);
        }
    }
}
END_TEST

START_TEST(test_initialize_output_constraint)
{
    int outputSize = 2;
    struct Interval *output_interval = Interval_new(outputSize, 1);
    initialize_output_constraint("artifacts/test_outputs/example1.out",
                                 output_interval,
                                 outputSize);
    assert_float_close(output_interval->lower_matrix->data[0],
                       -0.3, 1e-6);
    assert_float_close(output_interval->lower_matrix->data[1],
                       -0.01, 1e-6);
    assert_float_close(output_interval->upper_matrix->data[0],
                       0.2, 1e-6);
    assert_float_close(output_interval->upper_matrix->data[1],
                       1.0, 1e-6);
    destroy_Interval(output_interval);
}
END_TEST

START_TEST(test_initialize_output_constraint_not_exist)
{
    int outputSize = 2;
    initialize_output_constraint("<INVALID>", NULL, outputSize);
}
END_TEST

Suite *nnet_suite(void)
{
    Suite *s = suite_create("nnet");

    TCase *tc_load_conv_network = tcase_create("load_conv_network");
    tcase_add_exit_test(tc_load_conv_network, test_load_conv_network_invalid_filename, 1);
    tcase_add_test(tc_load_conv_network, test_load_conv_network_example1);
    tcase_add_test(tc_load_conv_network, test_load_conv_network_example2);
    suite_add_tcase(s, tc_load_conv_network);

    TCase *tc_evaluate_conv = tcase_create("evaluate_conv");
    tcase_add_test(tc_evaluate_conv, test_evaluate_conv_example1);
    tcase_add_test(tc_evaluate_conv, test_evaluate_conv_example2);
    suite_add_tcase(s, tc_evaluate_conv);

    TCase *tc_sort = tcase_create("sort");
    tcase_add_test(tc_sort, test_sort);
    tcase_add_test(tc_sort, test_sort_layers);
    suite_add_tcase(s, tc_sort);

    TCase *tc_inputs = tcase_create("inputs");
    tcase_add_test(tc_inputs, test_initialize_input_interval);
    tcase_add_test(tc_inputs, test_load_inputs);
    tcase_add_exit_test(tc_inputs, test_load_inputs_not_exist, 1);
    suite_add_tcase(s, tc_inputs);

    TCase *tc_outputs = tcase_create("outputs");
    tcase_add_test(tc_outputs, test_initialize_output_constraint);
    tcase_add_exit_test(tc_outputs, test_initialize_output_constraint_not_exist, 1);
    suite_add_tcase(s, tc_outputs);

    return s;
}