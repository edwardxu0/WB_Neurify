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

#include "split.h"

int MAX_THREAD = 0;
int NEED_PRINT = 0;

int adv_found = 0;
int count = 0;

int progress_list[PROGRESS_DEPTH];
int total_progress[PROGRESS_DEPTH];

int element_is_in(struct Matrix *x, struct Interval *interval)
{
    int d = fmax(interval->lower_matrix->row, interval->lower_matrix->col);
    for (int i = 0; i < d; i++)
    {
        if (x->data[i] < interval->lower_matrix->data[i])
        {
            return 0;
        }
        else if (x->data[i] > interval->upper_matrix->data[i])
        {
            return 0;
        }
    }
    return 1;
}

int interval_is_in(struct Interval *x, struct Interval *interval)
{
    return (element_is_in(x->lower_matrix, interval) &&
            element_is_in(x->upper_matrix, interval));
}

int check(struct NNet *nnet, struct Interval *output)
{
    int result = !interval_is_in(output, nnet->output_constraint);
    return result;
}

int check1(struct NNet *nnet, struct Matrix *output)
{
    int result = !element_is_in(output, nnet->output_constraint);
    return result;
}

void check_adv1(struct NNet *nnet, struct Matrix *adv)
{
    struct Matrix *output = Matrix_new(nnet->outputSize, 1);
    evaluate_conv(nnet, adv, output);
    int is_adv = check1(nnet, output);
    if (is_adv)
    {
        printf("adv found:\n");
        printMatrix(output);
        pthread_mutex_lock(&lock);
        adv_found = 1;
        pthread_mutex_unlock(&lock);
    }
    destroy_Matrix(output);
}

int sym_relu_lp(struct SymInterval *sInterval,
                struct Interval *input,
                struct NNet *nnet,
                int layer,
                int err_row,
                int *wrong_node_length,
                int *node_cnt,
                int target,
                int sig,
                int *sigs,
                lprec *lp)
{
    float tempVal_upper = 0.0, tempVal_lower = 0.0;

    int inputSize = nnet->inputSize;
    //record the number of wrong nodes
    int wcnt = 0;
    for (int i = 0; i < nnet->layerSizes[layer + 1]; i++)
    {
        relu_bound(sInterval, nnet, input, i, layer, err_row,
                   &tempVal_lower, &tempVal_upper);

        if (*node_cnt == target)
        {
            if (sig == 1)
            {
                set_node_constraints(lp,
                                     (*sInterval->new_eq_matrix).data,
                                     i * (inputSize + 1),
                                     sig,
                                     inputSize);
            }
            else
            {
                set_node_constraints(lp,
                                     (*sInterval->new_eq_matrix).data,
                                     i * (inputSize + 1),
                                     sig,
                                     inputSize);
                for (int k = 0; k < inputSize + 1; k++)
                {
                    (*sInterval->new_eq_matrix).data[k + i * (inputSize + 1)] = 0;
                }
                if (err_row > 0)
                {
                    for (int err_ind = 0; err_ind < err_row; err_ind++)
                    {
                        (*sInterval->new_err_matrix).data[err_ind + i * ERR_NODE] = 0;
                    }
                }
            }
            *node_cnt += 1;
            continue;
        }

        //Perform ReLU relaxation
        // handle the nodes that are split
        if (sigs[*node_cnt] == 0 && *node_cnt != target)
        {
            for (int k = 0; k < inputSize + 1; k++)
            {
                (*sInterval->new_eq_matrix).data[k + i * (inputSize + 1)] = 0;
            }
            if (err_row > 0)
            {
                for (int err_ind = 0; err_ind < err_row; err_ind++)
                {
                    (*sInterval->new_err_matrix).data[err_ind + i * ERR_NODE] = 0;
                }
            }
            *node_cnt += 1;
            continue;
        }

        if (sigs[*node_cnt] == 1 && *node_cnt != target)
        {
            *node_cnt += 1;
            continue;
        }

        if (tempVal_upper <= 0.0)
        {
            tempVal_upper = 0.0;
            for (int k = 0; k < inputSize + 1; k++)
            {
                (*sInterval->new_eq_matrix).data[k + i * (inputSize + 1)] = 0;
            }
            if (err_row > 0)
            {
                for (int err_ind = 0; err_ind < err_row; err_ind++)
                {
                    (*sInterval->new_err_matrix).data[err_ind + i * ERR_NODE] = 0;
                }
            }
        }
        else if (tempVal_lower >= 0.0)
        {
        }
        else
        {
            *wrong_node_length += 1;
            wcnt += 1;

            for (int k = 0; k < inputSize + 1; k++)
            {
                (*sInterval->new_eq_matrix).data[k + i * (inputSize + 1)] =
                    (*sInterval->new_eq_matrix).data[k + i * (inputSize + 1)] *
                    tempVal_upper / (tempVal_upper - tempVal_lower);
            }
            if (err_row > 0)
            {
                for (int err_ind = 0; err_ind < err_row; err_ind++)
                {
                    (*sInterval->new_err_matrix).data[err_ind + i * ERR_NODE] *=
                        tempVal_upper / (tempVal_upper - tempVal_lower);
                }
            }

            (*sInterval->new_err_matrix).data[*wrong_node_length - 1 + i * ERR_NODE] -=
                tempVal_upper * tempVal_lower /
                (tempVal_upper - tempVal_lower);
        }
        *node_cnt += 1;
    }

    return wcnt;
}

int forward_prop_interval_equation_conv_lp(struct NNet *nnet,
                                           struct Interval *input,
                                           struct SymInterval *sInterval,
                                           int *sigs,
                                           int target,
                                           int sig,
                                           lprec *lp)
{
    int node_cnt = 0;
    int need_to_split = 0;

    int numLayers = nnet->numLayers;
    int inputSize = nnet->inputSize;
    int outputSize = nnet->outputSize;
    int maxLayerSize = nnet->maxLayerSize;

    float *equation = sInterval->eq_matrix->data;
    float *new_equation = sInterval->new_eq_matrix->data;
    float *equation_err = sInterval->err_matrix->data;
    float *new_equation_err = sInterval->new_err_matrix->data;

    // equation is the temp equation for each layer
    memset(equation, 0, sizeof(float) * (inputSize + 1) * maxLayerSize);
    memset(equation_err, 0, sizeof(float) * ERR_NODE * maxLayerSize);

    float tempVal_upper = 0.0, tempVal_lower = 0.0;

    for (int i = 0; i < inputSize; i++)
    {
        equation[i * (inputSize + 1) + i] = 1;
    }

    //err_row is the number that is wrong before current layer
    int err_row = 0;
    int wrong_node_length = 0;
    for (int layer = 0; layer < numLayers; layer++)
    {
        memset(new_equation, 0, sizeof(float) * (inputSize + 1) * maxLayerSize);
        memset(new_equation_err, 0, sizeof(float) * ERR_NODE * maxLayerSize);

        if (nnet->layerTypes[layer] == 0)
        {
            sym_fc_layer(sInterval, nnet, layer, err_row);
        }
        else
        {
            sym_conv_layer(sInterval, nnet, layer, err_row);
        }

        if (layer < (numLayers - 1))
        {
            sym_relu_lp(sInterval,
                        input,
                        nnet,
                        layer,
                        err_row,
                        &wrong_node_length,
                        &node_cnt,
                        target,
                        sig,
                        sigs,
                        lp);
        }
        else
        {
            for (int i = 0; i < outputSize; i++)
            {
                if (NEED_PRINT)
                {
                    relu_bound(sInterval, nnet, input, i, layer, err_row,
                               &tempVal_lower, &tempVal_upper);
                    printf("target:%d, sig:%d, node:%d, l:%f, u:%f\n",
                           target, sig, i, tempVal_lower, tempVal_upper);
                }

                float upper_err = 0, lower_err = 0;
                for (int err_ind = 0; err_ind < err_row; err_ind++)
                {
                    float err = new_equation_err[err_ind + i * ERR_NODE];
                    if (err > 0)
                    {
                        upper_err += err;
                    }
                    else
                    {
                        lower_err += err;
                    }
                }

                float objective_value;
                struct Matrix *input_matrix = Matrix_new(1, inputSize);
                int isUnsat = 0;
                float lower_bound = nnet->output_constraint->lower_matrix->data[i];
                float upper_bound = nnet->output_constraint->upper_matrix->data[i];

                new_equation[inputSize + i * (inputSize + 1)] += lower_err;
                new_equation[inputSize + i * (inputSize + 1)] -= lower_bound;
                isUnsat = set_output_constraints(lp,
                                                 new_equation,
                                                 i * (inputSize + 1),
                                                 inputSize,
                                                 0,
                                                 &objective_value,
                                                 input_matrix->data);
                if (!isUnsat)
                {
                    need_to_split = 1;
                    if (NEED_PRINT)
                    {
                        printf("target:%d, sig:%d, node:%d--Objective value: %f\n",
                               target, sig, i, objective_value);
                    }
                    check_adv1(nnet, input_matrix);
                    if (adv_found)
                    {
                        return 0;
                    }
                }
                new_equation[inputSize + i * (inputSize + 1)] -= lower_err;
                new_equation[inputSize + i * (inputSize + 1)] += upper_err;
                new_equation[inputSize + i * (inputSize + 1)] += lower_bound;
                new_equation[inputSize + i * (inputSize + 1)] -= upper_bound;
                isUnsat = set_output_constraints(lp,
                                                 new_equation,
                                                 i * (inputSize + 1),
                                                 inputSize,
                                                 1,
                                                 &objective_value,
                                                 input_matrix->data);
                if (!isUnsat)
                {
                    need_to_split = 1;
                    if (NEED_PRINT)
                    {
                        printf("target:%d, sig:%d, node:%d--Objective value: %f\n",
                               target, sig, i, objective_value);
                    }
                    check_adv1(nnet, input_matrix);
                    if (adv_found)
                    {
                        return 0;
                    }
                }

                node_cnt++;
            }
        }
        memcpy(equation, new_equation, sizeof(float) * (inputSize + 1) * maxLayerSize);
        memcpy(equation_err, new_equation_err, sizeof(float) * (ERR_NODE)*maxLayerSize);
        sInterval->eq_matrix->row = sInterval->new_eq_matrix->row;
        sInterval->eq_matrix->col = sInterval->new_eq_matrix->col;
        sInterval->err_matrix->row = sInterval->new_err_matrix->row;
        sInterval->err_matrix->col = sInterval->new_err_matrix->col;
        err_row = wrong_node_length;
    }

    return need_to_split;
}

int direct_run_check_conv_lp(struct NNet *nnet,
                             struct Interval *input,
                             struct SymInterval *sInterval,
                             int *wrong_nodes,
                             int wrong_node_length,
                             int *sigs,
                             int target,
                             int sig,
                             lprec *lp,
                             int depth)
{
    pthread_mutex_lock(&lock);
    if (adv_found)
    {
        pthread_mutex_unlock(&lock);
        return 0;
    }
    pthread_mutex_unlock(&lock);

    int isOverlap = forward_prop_interval_equation_conv_lp(nnet,
                                                           input,
                                                           sInterval,
                                                           sigs,
                                                           target,
                                                           sig,
                                                           lp);
    if (depth <= PROGRESS_DEPTH && !isOverlap)
    {
        pthread_mutex_lock(&lock);
        progress_list[depth - 1] += 1;
        pthread_mutex_unlock(&lock);
        fprintf(stderr, " progress: ");
        for (int p = 1; p < PROGRESS_DEPTH + 1; p++)
        {
            if (p > depth)
            {
                total_progress[p - 1] -= pow(2, (p - depth));
            }
            fprintf(stderr, " %d/%d ", progress_list[p - 1], total_progress[p - 1]);
        }
        fprintf(stderr, "\n");
    }

    if (isOverlap)
    {
        if (NEED_PRINT)
            printf("depth:%d, sig:%d Need to split!\n\n", depth, sig);
        isOverlap = split_interval_conv_lp(nnet,
                                           input,
                                           sInterval,
                                           wrong_nodes,
                                           wrong_node_length,
                                           sigs,
                                           lp,
                                           depth);
    }
    else
    {
        if (!adv_found)
            if (NEED_PRINT)
                printf("depth:%d, sig:%d, UNSAT, great!\n\n", depth, sig);
    }
    return isOverlap;
}

/*
 * Multithread function
 */
void *direct_run_check_conv_lp_thread(void *args)
{
    struct direct_run_check_conv_lp_args *actual_args = args;
    direct_run_check_conv_lp(actual_args->nnet, actual_args->input,
                             actual_args->sym_interval,
                             actual_args->wrong_nodes,
                             actual_args->wrong_node_length,
                             actual_args->sigs,
                             actual_args->target,
                             actual_args->sig,
                             actual_args->lp,
                             actual_args->depth);
    return NULL;
}

int split_interval_conv_lp(struct NNet *nnet,
                           struct Interval *input,
                           struct SymInterval *sInterval,
                           int *wrong_nodes,
                           int wrong_node_length,
                           int *sigs,
                           lprec *lp,
                           int depth)
{
    pthread_mutex_lock(&lock);
    if (adv_found)
    {
        pthread_mutex_unlock(&lock);
        return 0;
    }

    if (depth >= wrong_node_length)
    {
        pthread_mutex_unlock(&lock);
        return 0;
    }
    pthread_mutex_unlock(&lock);

    if (depth == 0)
    {
        memset(progress_list, 0, PROGRESS_DEPTH * sizeof(int));
        for (int p = 1; p < PROGRESS_DEPTH + 1; p++)
        {
            total_progress[p - 1] = pow(2, p);
        }
    }

    int inputSize = nnet->inputSize;
    int maxLayerSize = nnet->maxLayerSize;

    int target = wrong_nodes[depth];
    depth++;

    int isOverlap1, isOverlap2;

    struct SymInterval *sInterval1 = SymInterval_new(inputSize, maxLayerSize, ERR_NODE);

    lprec *lp1, *lp2;
    lp1 = copy_lp(lp);
    lp2 = copy_lp(lp);

    int sigSize = 0;
    for (int layer = 1; layer < nnet->numLayers; layer++)
    {
        sigSize += nnet->layerSizes[layer];
    }

    int sigs1[sigSize];
    int sigs2[sigSize];
    memcpy(sigs1, sigs, sizeof(int) * sigSize);
    memcpy(sigs2, sigs, sizeof(int) * sigSize);

    int sig1, sig2;
    sig1 = 1;
    sig2 = 0;
    sigs1[target] = 1;
    sigs2[target] = 0;
    pthread_mutex_lock(&lock);
    if (count + 2 <= MAX_THREAD)
    {
        pthread_mutex_unlock(&lock);
        pthread_t workers1, workers2;
        struct direct_run_check_conv_lp_args args1 = {
            nnet,
            input,
            sInterval,
            wrong_nodes,
            wrong_node_length,
            sigs1,
            target,
            sig1,
            lp1,
            depth};

        struct direct_run_check_conv_lp_args args2 = {
            nnet,
            input,
            sInterval1,
            wrong_nodes,
            wrong_node_length,
            sigs2,
            target,
            sig2,
            lp2,
            depth};

        pthread_create(&workers1, NULL,
                       direct_run_check_conv_lp_thread, &args1);
        pthread_mutex_lock(&lock);
        count++;
        pthread_mutex_unlock(&lock);

        pthread_create(&workers2, NULL,
                       direct_run_check_conv_lp_thread, &args2);
        pthread_mutex_lock(&lock);
        count++;
        pthread_mutex_unlock(&lock);

        pthread_join(workers1, NULL);
        pthread_mutex_lock(&lock);
        count--;
        pthread_mutex_unlock(&lock);

        pthread_join(workers2, NULL);
        pthread_mutex_lock(&lock);
        count--;
        pthread_mutex_unlock(&lock);

        isOverlap1 = 0;
        isOverlap2 = 0;
    }
    else
    {
        pthread_mutex_unlock(&lock);
        isOverlap1 = direct_run_check_conv_lp(nnet,
                                              input,
                                              sInterval,
                                              wrong_nodes,
                                              wrong_node_length,
                                              sigs1,
                                              target,
                                              sig1,
                                              lp1,
                                              depth);

        isOverlap2 = direct_run_check_conv_lp(nnet,
                                              input,
                                              sInterval1,
                                              wrong_nodes,
                                              wrong_node_length,
                                              sigs2,
                                              target,
                                              sig2,
                                              lp2,
                                              depth);
    }

    destroy_SymInterval(sInterval1);
    delete_lp(lp1);
    delete_lp(lp2);

    int result = isOverlap1 || isOverlap2;
    depth--;

    if (!result && depth <= PROGRESS_DEPTH)
    {
        pthread_mutex_lock(&lock);
        progress_list[depth - 1] += 1;
        fprintf(stderr, " progress: ");
        for (int p = 1; p < PROGRESS_DEPTH + 1; p++)
        {
            fprintf(stderr, " %d/%d ",
                    progress_list[p - 1], total_progress[p - 1]);
        }
        fprintf(stderr, "\n");
        pthread_mutex_unlock(&lock);
    }

    return result;
}
