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

START_TEST(test_evaluate_conv_convolution)
{
    struct NNet *nnet = load_conv_network("artifacts/conv1.nnet");

    struct Matrix *input = Matrix_new(1, nnet->inputSize);
    struct Matrix *output = Matrix_new(nnet->outputSize, 1);

    float expected_output[nnet->outputSize];
    expected_output[0] = 225.0;
    expected_output[1] = 381.0;
    expected_output[2] = 405.0;
    expected_output[3] = 285.0;
    expected_output[4] = 502.0;
    expected_output[5] = 787.0;
    expected_output[6] = 796.0;
    expected_output[7] = 532.0;
    expected_output[8] = 550.0;
    expected_output[9] = 823.0;
    expected_output[10] = 832.0;
    expected_output[11] = 532.0;
    expected_output[12] = 301.0;
    expected_output[13] = 429.0;
    expected_output[14] = 417.0;
    expected_output[15] = 249.0;

    for (int i = 0; i < nnet->inputSize; i++)
    {
        input->data[i] = (i + 1) - 16;
    }

    evaluate_conv(nnet, input, output);
    for (int i = 0; i < nnet->outputSize; i++)
    {
        assert_float_close(output->data[i], expected_output[i], 1e-6);
    }

    destroy_conv_network(nnet);
    destroy_Matrix(input);
    destroy_Matrix(output);

    nnet = load_conv_network("artifacts/conv2.nnet");

    input = Matrix_new(1, nnet->inputSize);
    output = Matrix_new(nnet->outputSize, 1);

    float expected_output2[nnet->outputSize];
    expected_output2[0] = 1597.0;
    expected_output2[1] = 1615.0;
    expected_output2[2] = 1120.0;
    expected_output2[3] = 1705.0;
    expected_output2[4] = 1723.0;
    expected_output2[5] = 1120.0;
    expected_output2[6] = 903.0;
    expected_output2[7] = 879.0;
    expected_output2[8] = 513.0;

    for (int i = 0; i < nnet->inputSize; i++)
    {
        input->data[i] = (i + 1) - 36;
    }

    evaluate_conv(nnet, input, output);
    for (int i = 0; i < nnet->outputSize; i++)
    {
        assert_float_close(output->data[i], expected_output2[i], 1e-6);
    }

    destroy_conv_network(nnet);
    destroy_Matrix(input);
    destroy_Matrix(output);
}
END_TEST

START_TEST(test_evaluate_vs_forward)
{
    struct NNet *nnet = load_conv_network("artifacts/example2.nnet");
    int inputSize = nnet->inputSize;
    int outputSize = nnet->outputSize;

    struct Matrix *input = Matrix_new(1, inputSize);
    struct Matrix *output = Matrix_new(outputSize, 1);

    struct Interval *output_interval = Interval_new(outputSize, 1);

    int num_nodes = 0;
    for (int l = 1; l < nnet->numLayers; l++)
    {
        num_nodes += nnet->layerSizes[l];
    }
    ERR_NODE = num_nodes;
    struct Interval *input_interval = Interval_new(1, inputSize);
    int wrong_nodes[num_nodes];

    for (int j = 0; j < 15; j++)
    {
        int wrong_node_length = 0;
        for (int i = 0; i < inputSize; i++)
        {
            input->data[i] = (float)rand() / (float)(RAND_MAX / 1000.0);
        }
        initialize_input_interval(input_interval, inputSize, input->data, 0.0);
        forward_prop_interval_equation_linear_conv(
            nnet,
            input_interval,
            output_interval,
            NULL,
            wrong_nodes,
            &wrong_node_length);
        evaluate_conv(nnet, input, output);
        for (int i = 0; i < outputSize; i++)
        {
            assert_float_close(output->data[i], output_interval->lower_matrix->data[i], 1e-6);
            assert_float_close(output->data[i], output_interval->upper_matrix->data[i], 1e-6);
        }
    }

    destroy_Interval(input_interval);
    destroy_Interval(output_interval);
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
    destroy_Interval(interval);
}
END_TEST

START_TEST(test_initialize_interval_constraint)
{
    int outputSize = 2;
    struct Interval *output_interval = Interval_new(outputSize, 1);
    initialize_interval_constraint("artifacts/test_outputs/example1.out",
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

START_TEST(test_initialize_interval_constraint_not_exist)
{
    int outputSize = 2;
    initialize_interval_constraint("<INVALID>", NULL, outputSize);
}
END_TEST

START_TEST(test_initialize_hpoly_constraint)
{
    int inputSize = 2;
    struct HPoly *hpoly = HPoly_new(inputSize, 0);
    initialize_hpoly_constraint("artifacts/test_inputs/example.hpoly", hpoly);

    assert_float_close(hpoly->A->data[0], 1.0, 1e-6);
    assert_float_close(hpoly->A->data[1], 0.0, 1e-6);
    assert_float_close(hpoly->A->data[2], 0.0, 1e-6);
    assert_float_close(hpoly->A->data[3], 1.0, 1e-6);
    assert_float_close(hpoly->A->data[4], -1.0, 1e-6);
    assert_float_close(hpoly->A->data[5], 0.0, 1e-6);
    assert_float_close(hpoly->A->data[6], 0.0, 1e-6);
    assert_float_close(hpoly->A->data[7], -1.0, 1e-6);

    assert_float_close(hpoly->b->data[0], 1.0, 1e-6);
    assert_float_close(hpoly->b->data[1], 1.0, 1e-6);
    assert_float_close(hpoly->b->data[2], 0.0, 1e-6);
    assert_float_close(hpoly->b->data[3], 0.0, 1e-6);

    ck_assert_int_eq(hpoly->num_vars, 2);
    ck_assert_int_eq(hpoly->num_constraints, 4);

    destroy_HPoly(hpoly);
}
END_TEST

START_TEST(test_initialize_hpoly_constraint_not_exist)
{
    initialize_hpoly_constraint("<INVALID>", NULL);
}
END_TEST

START_TEST(test_set_hpoly_input_constraints)
{
    {
        int inputSize = 2;
        struct HPoly *hpoly = HPoly_new(inputSize, 0);
        initialize_hpoly_constraint("artifacts/test_inputs/example.hpoly", hpoly);

        lprec *lp = make_lp(0, inputSize);
        set_verbose(lp, CRITICAL);
        set_hpoly_input_constraints(hpoly, lp, inputSize);
        ck_assert_int_eq(solve(lp), 0);

        REAL row[inputSize + 1];
        get_variables(lp, row);
        for (int j = 0; j < inputSize; j++)
        {
            ck_assert(row[j] >= 0);
            ck_assert(row[j] <= 1);
        }

        destroy_HPoly(hpoly);
        delete_lp(lp);
    }
}
END_TEST

START_TEST(test_set_input_constraints)
{
    {
        int inputSize = 5;
        float input[5] = {0.2, 0.5, 0.6, 0.8, 0.8};
        float epsilon = 0.1;
        struct Interval *input_interval = Interval_new(1, inputSize);
        initialize_input_interval(input_interval, inputSize, input, epsilon);
        lprec *lp = make_lp(0, inputSize);
        set_verbose(lp, CRITICAL);
        set_input_constraints(input_interval, lp, inputSize);
        ck_assert_int_eq(solve(lp), 0);
        destroy_Interval(input_interval);
        delete_lp(lp);
    }

    {
        int inputSize = 5;
        float input[5] = {0.2, 0.5, 0.6, 0.8, 0.8};
        float epsilon = 1.1;
        struct Interval *input_interval = Interval_new(1, inputSize);
        initialize_input_interval(input_interval, inputSize, input, epsilon);
        lprec *lp = make_lp(0, inputSize);
        set_verbose(lp, CRITICAL);
        set_input_constraints(input_interval, lp, inputSize);
        ck_assert_int_eq(solve(lp), 0);
        destroy_Interval(input_interval);
        delete_lp(lp);
    }

    {
        int inputSize = 5;
        float input[5] = {-1.0, -0.5, 0.6, 0.8, -0.1};
        float epsilon = 0.1;
        struct Interval *input_interval = Interval_new(1, inputSize);
        initialize_input_interval(input_interval, inputSize, input, epsilon);
        lprec *lp = make_lp(0, inputSize);
        set_verbose(lp, CRITICAL);
        set_input_constraints(input_interval, lp, inputSize);
        ck_assert_int_eq(solve(lp), 0);
        destroy_Interval(input_interval);
        delete_lp(lp);
    }
}
END_TEST

START_TEST(test_forward_prop_interval_equation_linear_conv_example1)
{
    struct NNet *nnet = load_conv_network("artifacts/example1.nnet");
    int inputSize = nnet->inputSize;
    int outputSize = nnet->outputSize;

    float input[inputSize];
    load_inputs("artifacts/test_inputs/example1.center", inputSize, input);
    struct Interval *input_interval = Interval_new(1, inputSize);
    initialize_input_interval(input_interval, inputSize, input, 1);

    struct Interval *output_interval = Interval_new(outputSize, 1);

    int num_nodes = 4;
    ERR_NODE = num_nodes;
    int wrong_nodes[num_nodes];
    int wrong_node_length = 0;

    forward_prop_interval_equation_linear_conv(
        nnet,
        input_interval,
        output_interval,
        NULL,
        wrong_nodes,
        &wrong_node_length);
    ck_assert_int_eq(wrong_node_length, num_nodes);
    ck_assert(output_interval->lower_matrix->data[0] <= 1.0);
    ck_assert(output_interval->lower_matrix->data[1] <= 0.0);
    ck_assert(output_interval->upper_matrix->data[0] >= 5.0);
    ck_assert(output_interval->upper_matrix->data[1] >= 2.0);

    destroy_Interval(output_interval);

    float grad[wrong_node_length];
    wrong_node_length = 0;
    output_interval = Interval_new(outputSize, 1);
    forward_prop_interval_equation_linear_conv(
        nnet,
        input_interval,
        output_interval,
        grad,
        wrong_nodes,
        &wrong_node_length);
    ck_assert_int_eq(wrong_node_length, num_nodes);
    ck_assert(output_interval->lower_matrix->data[0] <= 1.0);
    ck_assert(output_interval->lower_matrix->data[1] <= 0.0);
    ck_assert(output_interval->upper_matrix->data[0] >= 5.0);
    ck_assert(output_interval->upper_matrix->data[1] >= 2.0);

    destroy_Interval(output_interval);

    wrong_node_length = 0;
    output_interval = Interval_new(outputSize, 1);
    for (int i = 0; i < inputSize; i++)
    {
        input[i] = -1;
    }
    initialize_input_interval(input_interval, inputSize, input, 0.00);
    forward_prop_interval_equation_linear_conv(
        nnet,
        input_interval,
        output_interval,
        NULL,
        wrong_nodes,
        &wrong_node_length);
    ck_assert_int_eq(wrong_node_length, 0);
    ck_assert(output_interval->lower_matrix->data[0] <= 1.0);
    ck_assert(output_interval->lower_matrix->data[1] <= 0.0);
    ck_assert(output_interval->upper_matrix->data[0] >= 1.0);
    ck_assert(output_interval->upper_matrix->data[1] >= 0.0);

    destroy_Interval(output_interval);

    wrong_node_length = 0;
    output_interval = Interval_new(outputSize, 1);
    for (int i = 0; i < inputSize; i++)
    {
        input[i] = -1;
    }
    initialize_input_interval(input_interval, inputSize, input, 0.00);
    forward_prop_interval_equation_linear_conv(
        nnet,
        input_interval,
        output_interval,
        grad,
        wrong_nodes,
        &wrong_node_length);
    ck_assert_int_eq(wrong_node_length, 0);
    ck_assert(output_interval->lower_matrix->data[0] <= 1.0);
    ck_assert(output_interval->lower_matrix->data[1] <= 0.0);
    ck_assert(output_interval->upper_matrix->data[0] >= 1.0);
    ck_assert(output_interval->upper_matrix->data[1] >= 0.0);

    destroy_Interval(output_interval);

    wrong_node_length = 0;
    output_interval = Interval_new(outputSize, 1);
    for (int i = 0; i < inputSize; i++)
    {
        input[i] = 1;
    }
    initialize_input_interval(input_interval, inputSize, input, 0.00);
    forward_prop_interval_equation_linear_conv(
        nnet,
        input_interval,
        output_interval,
        NULL,
        wrong_nodes,
        &wrong_node_length);
    ck_assert_int_eq(wrong_node_length, 0);
    ck_assert(output_interval->lower_matrix->data[0] <= 5.0);
    ck_assert(output_interval->lower_matrix->data[1] <= 2.0);
    ck_assert(output_interval->upper_matrix->data[0] >= 5.0);
    ck_assert(output_interval->upper_matrix->data[1] >= 2.0);

    destroy_Interval(output_interval);

    wrong_node_length = 0;
    output_interval = Interval_new(outputSize, 1);
    for (int i = 0; i < inputSize; i++)
    {
        input[i] = 1;
    }
    initialize_input_interval(input_interval, inputSize, input, 0.00);
    forward_prop_interval_equation_linear_conv(
        nnet,
        input_interval,
        output_interval,
        grad,
        wrong_nodes,
        &wrong_node_length);
    ck_assert_int_eq(wrong_node_length, 0);
    ck_assert(output_interval->lower_matrix->data[0] <= 5.0);
    ck_assert(output_interval->lower_matrix->data[1] <= 2.0);
    ck_assert(output_interval->upper_matrix->data[0] >= 5.0);
    ck_assert(output_interval->upper_matrix->data[1] >= 2.0);

    destroy_Interval(input_interval);
    destroy_Interval(output_interval);
    destroy_conv_network(nnet);
}
END_TEST

START_TEST(test_forward_prop_interval_equation_linear_conv_example2)
{
    struct NNet *nnet = load_conv_network("artifacts/example2.nnet");
    int inputSize = nnet->inputSize;
    int outputSize = nnet->outputSize;

    float input[inputSize];
    load_inputs("artifacts/test_inputs/example2.center", inputSize, input);
    struct Interval *input_interval = Interval_new(1, inputSize);
    initialize_input_interval(input_interval, inputSize, input, 1);

    struct Interval *output_interval = Interval_new(outputSize, 1);

    int num_nodes = 30;
    ERR_NODE = num_nodes;
    int wrong_nodes[num_nodes];
    int wrong_node_length = 0;

    forward_prop_interval_equation_linear_conv(
        nnet,
        input_interval,
        output_interval,
        NULL,
        wrong_nodes,
        &wrong_node_length);
    ck_assert_int_eq(wrong_node_length, 30);
    ck_assert(output_interval->lower_matrix->data[0] <= 0.0);
    ck_assert(output_interval->upper_matrix->data[0] >= 2304.0);

    destroy_Interval(output_interval);

    float grad[wrong_node_length];
    wrong_node_length = 0;
    output_interval = Interval_new(outputSize, 1);
    forward_prop_interval_equation_linear_conv(
        nnet,
        input_interval,
        output_interval,
        grad,
        wrong_nodes,
        &wrong_node_length);
    ck_assert_int_eq(wrong_node_length, 30);
    ck_assert(output_interval->lower_matrix->data[0] <= 0.0);
    ck_assert(output_interval->upper_matrix->data[0] >= 2304.0);

    destroy_Interval(input_interval);
    destroy_Interval(output_interval);
    destroy_conv_network(nnet);
}
END_TEST

START_TEST(test_backward_prop)
{
    struct NNet *nnet = load_conv_network("artifacts/example1.nnet");

    float grad[4];
    int R[nnet->numLayers][nnet->maxLayerSize];
    for (int l = 0; l < nnet->numLayers; l++)
    {
        for (int i = 0; i < nnet->maxLayerSize; i++)
        {
            R[l][i] = 0;
        }
    }
    backward_prop_conv(nnet, grad, R);
    assert_float_close(grad[0], 0.0, 1e-6);
    assert_float_close(grad[1], 0.0, 1e-6);
    assert_float_close(grad[2], 1.0, 1e-6);
    assert_float_close(grad[3], 2.0, 1e-6);

    for (int l = 0; l < nnet->numLayers; l++)
    {
        for (int i = 0; i < nnet->maxLayerSize; i++)
        {
            R[l][i] = 1;
        }
    }
    backward_prop_conv(nnet, grad, R);
    assert_float_close(grad[0], 3.0, 1e-6);
    assert_float_close(grad[1], 2.0, 1e-6);
    assert_float_close(grad[2], 1.0, 1e-6);
    assert_float_close(grad[3], 2.0, 1e-6);

    destroy_conv_network(nnet);
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
    tcase_add_test(tc_evaluate_conv, test_evaluate_conv_convolution);
    tcase_add_test(tc_evaluate_conv, test_evaluate_vs_forward);
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
    tcase_add_test(tc_outputs, test_initialize_interval_constraint);
    tcase_add_exit_test(tc_outputs, test_initialize_interval_constraint_not_exist, 1);
    suite_add_tcase(s, tc_outputs);

    TCase *tc_hpoly = tcase_create("hpoly");
    tcase_add_test(tc_hpoly, test_initialize_hpoly_constraint);
    tcase_add_exit_test(tc_hpoly, test_initialize_hpoly_constraint_not_exist, 1);
    tcase_add_test(tc_hpoly, test_set_hpoly_input_constraints);
    suite_add_tcase(s, tc_hpoly);

    TCase *tc_input_constraints = tcase_create("input_constraints");
    tcase_add_test(tc_input_constraints, test_set_input_constraints);
    suite_add_tcase(s, tc_input_constraints);

    TCase *tc_forward_prop_interval = tcase_create("forward_prop_interval");
    tcase_add_test(tc_forward_prop_interval,
                   test_forward_prop_interval_equation_linear_conv_example1);
    tcase_add_test(tc_forward_prop_interval,
                   test_forward_prop_interval_equation_linear_conv_example2);
    suite_add_tcase(s, tc_forward_prop_interval);

    TCase *tc_backward_prop = tcase_create("backward_prop");
    tcase_add_test(tc_backward_prop, test_backward_prop);
    suite_add_tcase(s, tc_backward_prop);

    return s;
}