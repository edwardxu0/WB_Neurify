/*
 ------------------------------------------------------------------
 ** Top contributors:
 **   Shiqi Wang
 ** This file is part of the Neurify project.
 ** Copyright (c) 2018-2019 by the authors listed in the file LICENSE
 ** and their institutional affiliations.
 ** All rights reserved.
 -----------------------------------------------------------------
 */

#include <argp.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "interval.h"
#include "matrix.h"
#include "nnet.h"
#include "split.h"

static struct argp_option options[] = {
    {"network", 'n', "NETWORK", 0, "The network to verify"},
    {"input", 'x', "INPUT", 0, "The seed input for verification"},
    {"linf", 'i', "EPSILON", OPTION_ARG_OPTIONAL, "Verify robustness under the L-Inf distance metric"},
    {"output", 'o', "OUTPUT", 0, "The output bounds for verification"},
    {"gamma_lb", 'l', "LB", 0, "The output lower bound for verification"},
    {"gamma_ub", 'u', "UB", 0, "The output upper bound for verification"},
    {"gamma_static", 's', 0, 0, "The output bounds for verification should not depend on the original output"},
    {0, 0, 0, 0, "Logging options:"},
    {"verbose", 'v', 0, 0, "Produce verbose output"},
    {0}};

struct arguments
{
    char *network;
    char *input;
    char *output;
    int property;
    float epsilon;
    float gamma_lb;
    float gamma_ub;
    int gamma_static;
};

static error_t parse_opt(int key, char *arg, struct argp_state *state)
{
    struct arguments *arguments = state->input;

    switch (key)
    {
    case 'n':
        arguments->network = arg;
        break;
    case 'x':
        arguments->input = arg;
        break;
    case 'o':
        arguments->output = arg;
        break;
    case 'l':
        arguments->gamma_lb = atof(arg);
        break;
    case 'u':
        arguments->gamma_ub = atof(arg);
        break;
    case 's':
        arguments->gamma_static = 1;
        break;
    case 'i':
        if (arguments->property)
        {
            printf("More than 1 input constraint is not supported.\n");
            argp_usage(state);
        }
        arguments->property = 1;
        arguments->epsilon = (arg) ? atof(arg) : 1.0;
        break;
    case 'v':
        NEED_PRINT = 1;
        break;
    case ARGP_KEY_END:
    case ARGP_NO_ARGS:
        if (arguments->network == NULL || arguments->input == NULL)
        {
            argp_usage(state);
            return ARGP_ERR_UNKNOWN;
        }
        break;
    default:
        return ARGP_ERR_UNKNOWN;
    }
    return 0;
}

static struct argp argp = {options, parse_opt, 0, 0};

int main(int argc, char *argv[])
{
    struct arguments arguments;
    arguments.network = NULL;
    arguments.input = NULL;
    arguments.output = NULL;
    arguments.property = 0;
    arguments.epsilon = 0;
    arguments.gamma_lb = -INFINITY;
    arguments.gamma_ub = INFINITY;
    arguments.gamma_static = 0;
    argp_parse(&argp, argc, argv, 0, 0, &arguments);

    printf("network: %s\n", arguments.network);
    printf("input: %s\n", arguments.input);
    printf("output: %s\n", arguments.output);
    printf("property: %d\n", arguments.property);
    printf("epsilon: %f\n\n", arguments.epsilon);

    openblas_set_num_threads(1);
    gettimeofday(&start_time, NULL);

    srand((unsigned)time(NULL));
    double time_spent;

    struct NNet *nnet = load_conv_network(arguments.network);

    int numLayers = nnet->numLayers;
    int inputSize = nnet->inputSize;
    int outputSize = nnet->outputSize;
    int maxLayerSize = nnet->maxLayerSize;

    struct Matrix *input_matrix = Matrix_new(1, inputSize);
    struct Matrix *output_matrix = Matrix_new(outputSize, 1);
    struct Interval *input_interval = Interval_new(1, inputSize);
    struct Interval *output_interval = Interval_new(outputSize, 1);
    struct Interval *output_constraint = Interval_new(outputSize, 1);
    nnet->output_constraint = output_constraint;

    int wrong_node_length = 0;
    for (int layer = 1; layer < numLayers; layer++)
    {
        wrong_node_length += nnet->layerSizes[layer];
    }

    int wrong_nodes[wrong_node_length];
    memset(wrong_nodes, 0, sizeof(int) * wrong_node_length);
    float grad[wrong_node_length];
    memset(grad, 0, sizeof(float) * wrong_node_length);
    int sigs[wrong_node_length];
    for (int i = 0; i < wrong_node_length; i++)
    {
        sigs[i] = -1;
    }
    ERR_NODE = wrong_node_length;
    wrong_node_length = 0;

    load_inputs(arguments.input, inputSize, input_matrix->data);
    initialize_input_interval(input_interval,
                              inputSize,
                              input_matrix->data,
                              arguments.epsilon);

    evaluate_conv(nnet, input_matrix, output_matrix);
    if (inputSize < 10)
    {
        printf("concrete input:");
        printMatrix(input_matrix);
    }
    printf("concrete output:");
    printMatrix(output_matrix);
    if (arguments.property == 0)
        return 0;

    if (arguments.output != NULL)
    {
        initialize_output_constraint(arguments.output, output_constraint, outputSize);
    }
    else
    {
        for (int i = 0; i < outputSize; i++)
        {
            if (arguments.gamma_static)
            {
                output_constraint->upper_matrix->data[i] = arguments.gamma_ub;
                output_constraint->lower_matrix->data[i] = arguments.gamma_lb;
            }
            else
            {
                output_constraint->upper_matrix->data[i] = output_matrix->data[i] + arguments.gamma_ub;
                output_constraint->lower_matrix->data[i] = output_matrix->data[i] - fabs(arguments.gamma_lb);
            }
        }
    }
    printf("Output constraint:\n");
    printf("  lower bound:");
    printMatrix(nnet->output_constraint->lower_matrix);
    printf("  upper bound:");
    printMatrix(nnet->output_constraint->upper_matrix);

    int isOverlap = check1(nnet, output_matrix);
    if (!isOverlap)
    {
        struct SymInterval *sym_interval = SymInterval_new(inputSize, maxLayerSize, ERR_NODE);
        forward_prop_interval_equation_linear_conv(nnet,
                                                   input_interval,
                                                   output_interval,
                                                   grad,
                                                   sym_interval,
                                                   wrong_nodes,
                                                   &wrong_node_length);
        ERR_NODE = wrong_node_length;
        destroy_SymInterval(sym_interval);
        sym_interval = SymInterval_new(inputSize, maxLayerSize, ERR_NODE);

        printf("One shot approximation:\n");
        printf("upper_matrix:");
        printMatrix(output_interval->upper_matrix);
        printf("lower matrix:");
        printMatrix(output_interval->lower_matrix);

        sort(grad, wrong_node_length, wrong_nodes);
        sort_layers(nnet->numLayers, nnet->layerSizes,
                    wrong_node_length, wrong_nodes);

        printf("total wrong nodes: %d\n", wrong_node_length);

        isOverlap = check(nnet, output_interval);
        if (isOverlap)
        {
            printf("Regular Mode (No CHECK_ADV_MODE)\n");

            lprec *lp = make_lp(0, inputSize);
            set_verbose(lp, IMPORTANT);
            set_input_constraints(input_interval, lp, inputSize);
            set_presolve(lp, PRESOLVE_LINDEP, get_presolveloops(lp));

            int depth = 0;
            isOverlap = split_interval_conv_lp(nnet,
                                               input_interval,
                                               sym_interval,
                                               wrong_nodes,
                                               wrong_node_length,
                                               sigs,
                                               lp,
                                               depth);
            delete_lp(lp);
        }
        destroy_SymInterval(sym_interval);
    }
    else
    {
        adv_found = 1;
    }

    gettimeofday(&finish_time, NULL);
    time_spent = ((float)(finish_time.tv_sec - start_time.tv_sec) * 1000000 +
                  (float)(finish_time.tv_usec - start_time.tv_usec)) /
                 1000000;

    if (isOverlap == 0 && adv_found == 0 && !can_t_prove)
    {
        printf("Proved.\n");
    }
    else if (adv_found)
    {
        printf("Falsified.\n");
    }
    else
    {
        printf("Unknown.\n");
    }
    printf("time: %f\n", time_spent);

    destroy_conv_network(nnet);
    destroy_Matrix(input_matrix);
    destroy_Matrix(output_matrix);
    destroy_Interval(input_interval);
    destroy_Interval(output_interval);
    destroy_Interval(output_constraint);
}
