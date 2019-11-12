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

#include "nnet.h"

long long ERR_NODE = 0;

struct timeval start_time, finish_time;

struct SymInterval *SymInterval_new(int inputSize, int maxLayerSize, int maxErrNodes)
{
    struct SymInterval *sym_interval = malloc(sizeof(struct SymInterval));
    sym_interval->eq_matrix = Matrix_new((inputSize + 1), maxLayerSize);
    sym_interval->new_eq_matrix = Matrix_new((inputSize + 1), maxLayerSize);
    sym_interval->err_matrix = Matrix_new(maxErrNodes, maxLayerSize);
    sym_interval->new_err_matrix = Matrix_new(maxErrNodes, maxLayerSize);
    return sym_interval;
}

void destroy_SymInterval(struct SymInterval *sym_interval)
{
    destroy_Matrix(sym_interval->eq_matrix);
    destroy_Matrix(sym_interval->new_eq_matrix);
    destroy_Matrix(sym_interval->err_matrix);
    destroy_Matrix(sym_interval->new_err_matrix);
    free(sym_interval);
}

//Take in a .nnet filename with path and load the network from the file
//Inputs:  filename - const char* that specifies the name and path of file
//Outputs: void *   - points to the loaded neural network
struct NNet *load_conv_network(const char *filename)
{
    //Load file and check if it exists
    FILE *fstream = fopen(filename, "r");

    if (fstream == NULL)
    {
        printf("Wrong network!\n");
        exit(1);
    }

    //Initialize variables
    int bufferSize = 650000;
    char *buffer = (char *)malloc(sizeof(char) * bufferSize);
    char *record, *line;

    struct NNet *nnet = (struct NNet *)malloc(sizeof(struct NNet));

    //skip header lines
    line = fgets(buffer, bufferSize, fstream);
    while (strstr(line, "//") != NULL)
        line = fgets(buffer, bufferSize, fstream);

    //Read int parameters of neural network
    record = strtok(line, ",\n");
    nnet->numLayers = atoi(record);
    nnet->inputSize = atoi(strtok(NULL, ",\n"));
    nnet->outputSize = atoi(strtok(NULL, ",\n"));
    nnet->maxLayerSize = atoi(strtok(NULL, ",\n"));

    //Allocate space for and read values of the array members of the network
    nnet->layerSizes = (int *)malloc(sizeof(int) * (nnet->numLayers + 1));
    line = fgets(buffer, bufferSize, fstream);
    record = strtok(line, ",\n");
    for (int i = 0; i < (nnet->numLayers + 1); i++)
    {
        nnet->layerSizes[i] = atoi(record);
        record = strtok(NULL, ",\n");
    }

    nnet->layerTypes = (int *)malloc(sizeof(int) * nnet->numLayers);
    nnet->convLayersNum = 0;
    line = fgets(buffer, bufferSize, fstream);
    record = strtok(line, ",\n");
    for (int i = 0; i < nnet->numLayers; i++)
    {
        nnet->layerTypes[i] = atoi(record);
        if (nnet->layerTypes[i] == 1)
        {
            nnet->convLayersNum++;
        }
        record = strtok(NULL, ",\n");
    }

    //initial convlayer parameters
    nnet->convLayer = (int **)malloc(sizeof(int *) * nnet->convLayersNum);
    for (int i = 0; i < nnet->convLayersNum; i++)
    {
        nnet->convLayer[i] = (int *)malloc(sizeof(int) * 5);
    }

    for (int cl = 0; cl < nnet->convLayersNum; cl++)
    {
        line = fgets(buffer, bufferSize, fstream);
        record = strtok(line, ",\n");
        for (int i = 0; i < 5; i++)
        {
            nnet->convLayer[cl][i] = atoi(record);
            record = strtok(NULL, ",\n");
        }
        if (record != NULL)
        {
            nnet->convLayer[cl][4] += atoi(record);
        }
        else
        {
            nnet->convLayer[cl][4] *= 2;
        }
    }

    //Allocate space for matrix of Neural Network
    //
    //The first dimension will be the layer number
    //The second dimension will be 0 for weights, 1 for biases
    //The third dimension will be the number of neurons in that layer
    //The fourth dimension will be the number of inputs to that layer
    //
    //Note that the bias array will have only number per neuron, so
    //    its fourth dimension will always be one
    //
    nnet->matrix = (float ****)malloc(sizeof(float *) * (nnet->numLayers));
    for (int layer = 0; layer < nnet->numLayers; layer++)
    {
        if (nnet->layerTypes[layer] == 0)
        {
            nnet->matrix[layer] = (float ***)malloc(sizeof(float *) * 2);
            nnet->matrix[layer][0] = (float **)malloc(sizeof(float *) * nnet->layerSizes[layer + 1]);
            nnet->matrix[layer][1] = (float **)malloc(sizeof(float *) * nnet->layerSizes[layer + 1]);
            for (int row = 0; row < nnet->layerSizes[layer + 1]; row++)
            {
                nnet->matrix[layer][0][row] = (float *)malloc(sizeof(float) * nnet->layerSizes[layer]);
                nnet->matrix[layer][1][row] = (float *)malloc(sizeof(float));
            }
        }
    }

    nnet->conv_matrix = (float ****)malloc(sizeof(float *) * nnet->convLayersNum);
    for (int layer = 0; layer < nnet->convLayersNum; layer++)
    {
        int out_channel = nnet->convLayer[layer][0];
        int in_channel = nnet->convLayer[layer][1];
        int kernel_size = nnet->convLayer[layer][2] * nnet->convLayer[layer][2];
        nnet->conv_matrix[layer] = (float ***)malloc(sizeof(float *) * out_channel);
        for (int oc = 0; oc < out_channel; oc++)
        {
            nnet->conv_matrix[layer][oc] = (float **)malloc(sizeof(float *) * in_channel);
            for (int ic = 0; ic < in_channel; ic++)
            {
                nnet->conv_matrix[layer][oc][ic] = (float *)malloc(sizeof(float) * kernel_size);
            }
        }
    }

    nnet->conv_bias = (float **)malloc(sizeof(float *) * nnet->convLayersNum);
    for (int layer = 0; layer < nnet->convLayersNum; layer++)
    {
        int out_channel = nnet->convLayer[layer][0];
        nnet->conv_bias[layer] = (float *)malloc(sizeof(float) * out_channel);
    }

    {
        int layer = 0;
        int param = 0;
        int i = 0;
        int j = 0;
        char *tmpptr = NULL;

        int oc = 0, ic = 0, kernel = 0;
        int kernel_size = 0;

        //Read in parameters and put them in the matrix
        float w = 0.0;
        while ((line = fgets(buffer, bufferSize, fstream)) != NULL)
        {
            if (nnet->layerTypes[layer] == 1)
            {
                int out_channel = nnet->convLayer[layer][0];
                kernel_size = nnet->convLayer[layer][2] * nnet->convLayer[layer][2];
                if (oc >= out_channel)
                {
                    if (param == 0)
                    {
                        param = 1;
                    }
                    else
                    {
                        param = 0;
                        layer++;
                        if (nnet->layerTypes[layer] == 1)
                        {
                            out_channel = nnet->convLayer[layer][0];
                            kernel_size = nnet->convLayer[layer][2] * nnet->convLayer[layer][2];
                        }
                    }
                    oc = 0;
                    ic = 0;
                    kernel = 0;
                }
            }
            else
            {
                if (i >= nnet->layerSizes[layer + 1])
                {
                    if (param == 0)
                    {
                        param = 1;
                    }
                    else
                    {
                        param = 0;
                        layer++;
                    }
                    i = 0;
                    j = 0;
                }
            }

            if (nnet->layerTypes[layer] == 1)
            {
                if (param == 0)
                {
                    record = strtok_r(line, ",\n", &tmpptr);
                    while (record != NULL)
                    {
                        w = (float)atof(record);
                        nnet->conv_matrix[layer][oc][ic][kernel] = w;
                        kernel++;
                        if (kernel == kernel_size)
                        {
                            kernel = 0;
                            ic++;
                        }
                        record = strtok_r(NULL, ",\n", &tmpptr);
                    }
                    tmpptr = NULL;
                    kernel = 0;
                    ic = 0;
                    oc++;
                }
                else
                {
                    record = strtok_r(line, ",\n", &tmpptr);
                    while (record != NULL)
                    {
                        w = (float)atof(record);
                        nnet->conv_bias[layer][oc] = w;
                        record = strtok_r(NULL, ",\n", &tmpptr);
                    }
                    tmpptr = NULL;
                    oc++;
                }
            }
            else
            {
                record = strtok_r(line, ",\n", &tmpptr);
                while (record != NULL)
                {
                    w = (float)atof(record);
                    nnet->matrix[layer][param][i][j] = w;
                    j++;
                    record = strtok_r(NULL, ",\n", &tmpptr);
                }
                tmpptr = NULL;
                j = 0;
                i++;
            }
        }
    }

    struct Matrix *weights = malloc(nnet->numLayers * sizeof(struct Matrix));
    struct Matrix *bias = malloc(nnet->numLayers * sizeof(struct Matrix));

    for (int layer = 0; layer < nnet->numLayers; layer++)
    {
        if (nnet->layerTypes[layer] == 1)
            continue;

        weights[layer].row = nnet->layerSizes[layer];
        weights[layer].col = nnet->layerSizes[layer + 1];
        weights[layer].data = (float *)malloc(sizeof(float) * weights[layer].row * weights[layer].col);
        int n = 0;
        for (int i = 0; i < weights[layer].col; i++)
        {
            for (int j = 0; j < weights[layer].row; j++)
            {
                weights[layer].data[n] = nnet->matrix[layer][0][i][j];
                n++;
            }
        }
        bias[layer].col = nnet->layerSizes[layer + 1];
        bias[layer].row = (float)1;
        bias[layer].data = (float *)malloc(sizeof(float) * bias[layer].col);
        for (int i = 0; i < bias[layer].col; i++)
        {
            bias[layer].data[i] = nnet->matrix[layer][1][i][0];
        }
    }
    nnet->weights = weights;
    nnet->bias = bias;

    free(buffer);
    fclose(fstream);
    return nnet;
}

void destroy_conv_network(struct NNet *nnet)
{
    if (nnet != NULL)
    {
        for (int i = 0; i < nnet->numLayers; i++)
        {
            if (nnet->layerTypes[i] == 1)
                continue;
            for (int row = 0; row < nnet->layerSizes[i + 1]; row++)
            {
                //free weight and bias arrays
                free(nnet->matrix[i][0][row]);
                free(nnet->matrix[i][1][row]);
            }
            //free pointer to weights and biases
            free(nnet->matrix[i][0]);
            free(nnet->matrix[i][1]);
            free(nnet->weights[i].data);
            free(nnet->bias[i].data);
            free(nnet->matrix[i]);
        }
        for (int i = 0; i < nnet->convLayersNum; i++)
        {
            int in_channel = nnet->convLayer[i][1];
            int out_channel = nnet->convLayer[i][0];
            for (int oc = 0; oc < out_channel; oc++)
            {
                for (int ic = 0; ic < in_channel; ic++)
                {
                    free(nnet->conv_matrix[i][oc][ic]);
                }
                free(nnet->conv_matrix[i][oc]);
            }
            free(nnet->conv_matrix[i]);
            free(nnet->conv_bias[i]);
        }
        free(nnet->conv_bias);
        free(nnet->conv_matrix);
        for (int i = 0; i < nnet->convLayersNum; i++)
        {
            free(nnet->convLayer[i]);
        }
        free(nnet->convLayer);
        free(nnet->weights);
        free(nnet->bias);
        free(nnet->layerSizes);
        free(nnet->layerTypes);
        free(nnet->matrix);
        free(nnet);
    }
}

void sort(float *array, int num, int *ind)
{
    for (int i = 0; i < num; i++)
    {
        int index = i;
        for (int j = i + 1; j < num; j++)
        {
            if (array[ind[j]] > array[ind[index]])
            {
                index = j;
            }
        }
        int tmp_ind = ind[i];
        ind[i] = ind[index];
        ind[index] = tmp_ind;
    }
}

void sort_layers(int numLayers, int *layerSizes, int wrong_node_length, int *wrong_nodes)
{
    int wrong_nodes_tmp[wrong_node_length];
    memset(wrong_nodes_tmp, 0, sizeof(int) * wrong_node_length);
    int j = 0;
    int count_node = 0;
    for (int layer = 1; layer < numLayers; layer++)
    {
        count_node += layerSizes[layer];
        for (int i = 0; i < wrong_node_length; i++)
        {
            if (wrong_nodes[i] < count_node && wrong_nodes[i] >= count_node - layerSizes[layer])
            {
                wrong_nodes_tmp[j] = wrong_nodes[i];
                j++;
            }
        }
    }
    memcpy(wrong_nodes, wrong_nodes_tmp, sizeof(int) * wrong_node_length);
}

void set_input_constraints(struct Interval *input,
                           lprec *lp,
                           int inputSize)
{
    for (int var = 1; var < inputSize + 1; var++)
    {
        set_bounds(lp, var, input->lower_matrix->data[var - 1], input->upper_matrix->data[var - 1]);
    }
}

void set_node_constraints(lprec *lp,
                          float *equation,
                          int start,
                          int sig,
                          int inputSize)
{
    REAL row[inputSize + 1];
    memset(row, 0, inputSize * sizeof(float));
    set_add_rowmode(lp, TRUE);
    for (int j = 1; j < inputSize + 1; j++)
    {
        row[j] = equation[start + j - 1];
    }
    if (sig == 1)
    {
        add_constraintex(lp, 1, row, NULL, GE, -equation[inputSize + start]);
    }
    else
    {
        add_constraintex(lp, 1, row, NULL, LE, -equation[inputSize + start]);
    }
    set_add_rowmode(lp, FALSE);
}

float set_output_constraints(lprec *lp,
                             float *equation,
                             int start_place,
                             int inputSize,
                             int is_max,
                             float *output,
                             float *input_prev)
{
    int unsat = 0;
    REAL row[inputSize + 1];
    memset(row, 0, inputSize * sizeof(float));
    set_add_rowmode(lp, TRUE);
    for (int j = 1; j < inputSize + 1; j++)
    {
        row[j] = equation[start_place + j - 1];
    }
    if (is_max)
    {
        add_constraintex(lp, 1, row, NULL, GE, -equation[inputSize + start_place]);
        set_maxim(lp);
    }
    else
    {
        add_constraintex(lp, 1, row, NULL, LE, -equation[inputSize + start_place]);
        set_minim(lp);
    }
    set_add_rowmode(lp, FALSE);

    set_obj_fnex(lp, inputSize + 1, row, NULL);

    int ret = solve(lp);
    if (ret == OPTIMAL)
    {
        *output = get_objective(lp) + equation[inputSize + start_place];
        get_variables(lp, row);
        for (int j = 0; j < inputSize; j++)
        {
            input_prev[j] = (float)row[j];
        }
    }
    else
    {
        unsat = 1;
    }

    del_constraint(lp, get_Nrows(lp));

    return unsat;
}

void initialize_input_interval(struct Interval *input_interval,
                               int inputSize,
                               float input[inputSize],
                               float epsilon)
{
    for (int i = 0; i < inputSize; i++)
    {
        input_interval->upper_matrix->data[i] = input[i] + epsilon;
        input_interval->lower_matrix->data[i] = input[i] - epsilon;
    }
}

void initialize_output_constraint(const char *path, struct Interval *output_interval, int outputSize)
{
    FILE *fstream = fopen(path, "r");
    if (fstream == NULL)
    {
        printf("no output:%s!\n", path);
        exit(1);
    }
    int bufferSize = 300000;
    char *buffer = (char *)malloc(sizeof(char) * bufferSize);
    char *record, *line;
    line = fgets(buffer, bufferSize, fstream);
    record = strtok(line, ",\n");
    for (int i = 0; i < outputSize; i++)
    {
        output_interval->lower_matrix->data[i] = atof(record);
        record = strtok(NULL, ",\n");
    }
    line = fgets(buffer, bufferSize, fstream);
    record = strtok(line, ",\n");
    for (int i = 0; i < outputSize; i++)
    {
        output_interval->upper_matrix->data[i] = atof(record);
        record = strtok(NULL, ",\n");
    }
    free(buffer);
    fclose(fstream);
}

void load_inputs(const char *input_path, int inputSize, float *input)
{
    FILE *fstream = fopen(input_path, "r");
    if (fstream == NULL)
    {
        printf("no input:%s!\n", input_path);
        exit(1);
    }
    int bufferSize = 300000;
    char *buffer = (char *)malloc(sizeof(char) * bufferSize);
    char *record, *line;
    line = fgets(buffer, bufferSize, fstream);
    record = strtok(line, ",\n");
    for (int i = 0; i < inputSize; i++)
    {

        input[i] = atof(record);
        record = strtok(NULL, ",\n");
    }
    free(buffer);
    fclose(fstream);
}

int evaluate_conv(struct NNet *network, struct Matrix *input, struct Matrix *output)
{
    struct NNet *nnet = network;
    int numLayers = nnet->numLayers;
    int inputSize = nnet->inputSize;
    int outputSize = nnet->outputSize;
    int maxLayerSize = nnet->maxLayerSize;

    float ****matrix = nnet->matrix;
    float ****conv_matrix = nnet->conv_matrix;

    float tempVal;
    float z[nnet->maxLayerSize];
    float a[nnet->maxLayerSize];

    for (int i = 0; i < inputSize; i++)
    {
        z[i] = input->data[i];
    }

    for (int layer = 0; layer < numLayers; layer++)
    {
        memset(a, 0, sizeof(float) * maxLayerSize);

        if (nnet->layerTypes[layer] == 0)
        {
            for (int i = 0; i < nnet->layerSizes[layer + 1]; i++)
            {
                float **weights = matrix[layer][0];
                float **biases = matrix[layer][1];
                tempVal = 0.0;

                //Perform weighted summation of inputs
                for (int j = 0; j < nnet->layerSizes[layer]; j++)
                {
                    tempVal += z[j] * weights[i][j];
                }

                //Add bias to weighted sum
                tempVal += biases[i][0];

                //Perform ReLU
                if (tempVal < 0.0 && layer < (numLayers - 1))
                {
                    tempVal = 0.0;
                }
                a[i] = tempVal;
            }
            for (int j = 0; j < maxLayerSize; j++)
            {
                z[j] = a[j];
            }
        }
        else
        {
            int out_channel = nnet->convLayer[layer][0];
            int in_channel = nnet->convLayer[layer][1];
            int kernel_size = nnet->convLayer[layer][2];
            int stride = nnet->convLayer[layer][3];
            int padding = nnet->convLayer[layer][4];
            int lt_pad = padding / 2;
            //size is the input size in each channel
            int size = sqrt(nnet->layerSizes[layer] / in_channel);
            //padding size is the input size after padding
            int padding_size = size + padding;
            //out_size is the output size in each channel after kernel
            int out_size = ceil((padding_size + 1.0 - kernel_size) / stride);

            float *z_new = (float *)malloc(sizeof(float) * padding_size * padding_size * in_channel);
            memset(z_new, 0, sizeof(float) * padding_size * padding_size * in_channel);
            for (int ic = 0; ic < in_channel; ic++)
            {
                for (int h = 0; h < size; h++)
                {
                    for (int w = 0; w < size; w++)
                    {
                        z_new[ic * padding_size * padding_size + padding_size * (h + lt_pad) + w + lt_pad] =
                            z[ic * size * size + size * h + w];
                    }
                }
            }

            for (int oc = 0; oc < out_channel; oc++)
            {
                for (int oh = 0; oh < out_size; oh++)
                {
                    for (int ow = 0; ow < out_size; ow++)
                    {
                        int start = ow * stride + oh * stride * padding_size;
                        for (int kh = 0; kh < kernel_size; kh++)
                        {
                            for (int kw = 0; kw < kernel_size; kw++)
                            {
                                for (int ic = 0; ic < in_channel; ic++)
                                {
                                    a[oc * out_size * out_size + oh * out_size + ow] +=
                                        conv_matrix[layer][oc][ic][kh * kernel_size + kw] *
                                        z_new[ic * padding_size * padding_size + padding_size * kh + kw + start];
                                }
                            }
                        }
                        a[oc * out_size * out_size + ow + oh * out_size] += nnet->conv_bias[layer][oc];
                    }
                }
            }
            for (int j = 0; j < maxLayerSize; j++)
            {
                if (a[j] < 0)
                {
                    a[j] = 0;
                }
                z[j] = a[j];
            }
            free(z_new);
        }
    }

    for (int i = 0; i < outputSize; i++)
    {
        output->data[i] = a[i];
    }

    return 1;
}

void backward_prop_conv(struct NNet *nnet, float *grad,
                        int R[][nnet->maxLayerSize])
{
    int numLayers = nnet->numLayers;
    int inputSize = nnet->inputSize;
    int maxLayerSize = nnet->maxLayerSize;

    float grad_upper[maxLayerSize];
    float grad_lower[maxLayerSize];
    float grad1_upper[maxLayerSize];
    float grad1_lower[maxLayerSize];
    memcpy(grad_upper, nnet->matrix[numLayers - 1][0][nnet->target],
           sizeof(float) * nnet->layerSizes[numLayers - 1]);
    memcpy(grad_lower, nnet->matrix[numLayers - 1][0][nnet->target],
           sizeof(float) * nnet->layerSizes[numLayers - 1]);
    {
        int start_node = 0;
        for (int l = 1; l < nnet->numLayers - 1; l++)
        {
            start_node += nnet->layerSizes[l];
        }
        for (int i = 0; i < nnet->layerSizes[nnet->numLayers - 1]; i++)
        {
            grad[start_node + i] = (grad_upper[i] > -grad_lower[i]) ? grad_upper[i] : -grad_lower[i];
        }
    }

    for (int layer = numLayers - 2; layer > -1; layer--)
    {
        float **weights = nnet->matrix[layer][0];
        memset(grad1_upper, 0, sizeof(float) * maxLayerSize);
        memset(grad1_lower, 0, sizeof(float) * maxLayerSize);

        if (nnet->layerTypes[layer] != 1)
        {
            if (layer != 0)
            {
                for (int j = 0; j < nnet->layerSizes[numLayers - 1]; j++)
                {
                    if (R[layer][j] == 0)
                    {
                        grad_upper[j] = grad_lower[j] = 0;
                    }
                    else if (R[layer][j] == 1)
                    {
                        grad_upper[j] = (grad_upper[j] > 0) ? grad_upper[j] : 0;
                        grad_lower[j] = (grad_lower[j] < 0) ? grad_lower[j] : 0;
                    }

                    for (int i = 0; i < nnet->layerSizes[numLayers - 1]; i++)
                    {
                        if (weights[j][i] >= 0)
                        {
                            grad1_upper[i] += weights[j][i] * grad_upper[j];
                            grad1_lower[i] += weights[j][i] * grad_lower[j];
                        }
                        else
                        {
                            grad1_upper[i] += weights[j][i] * grad_lower[j];
                            grad1_lower[i] += weights[j][i] * grad_upper[j];
                        }
                    }
                }
            }
            else
            {
                for (int j = 0; j < nnet->layerSizes[numLayers - 1]; j++)
                {
                    if (R[layer][j] == 0)
                    {
                        grad_upper[j] = grad_lower[j] = 0;
                    }
                    else if (R[layer][j] == 1)
                    {
                        grad_upper[j] = (grad_upper[j] > 0) ? grad_upper[j] : 0;
                        grad_lower[j] = (grad_lower[j] < 0) ? grad_lower[j] : 0;
                    }

                    for (int i = 0; i < inputSize; i++)
                    {
                        if (weights[j][i] >= 0)
                        {
                            grad1_upper[i] += weights[j][i] * grad_upper[j];
                            grad1_lower[i] += weights[j][i] * grad_lower[j];
                        }
                        else
                        {
                            grad1_upper[i] += weights[j][i] * grad_lower[j];
                            grad1_lower[i] += weights[j][i] * grad_upper[j];
                        }
                    }
                }
            }
        }
        else
        {
            break;
        }

        if (layer != 0 && nnet->layerTypes[layer - 1] != 1)
        {
            memcpy(grad_upper, grad1_upper, sizeof(float) * nnet->layerSizes[numLayers - 1]);
            memcpy(grad_lower, grad1_lower, sizeof(float) * nnet->layerSizes[numLayers - 1]);
            int start_node = 0;
            for (int l = 1; l < layer; l++)
            {
                start_node += nnet->layerSizes[l];
            }
            for (int i = 0; i < nnet->layerSizes[layer]; i++)
            {
                grad[start_node + i] = (grad_upper[i] > -grad_lower[i]) ? grad_upper[i] : -grad_lower[i];
            }
        }
        else
        {
            break;
        }
    }
}

void sym_fc_layer(struct SymInterval *sInterval, struct NNet *nnet,
                  int layer, int err_row)
{
    int inputSize = nnet->inputSize;
    int layerSize = nnet->layerSizes[layer + 1];

    struct Matrix weights = nnet->weights[layer];
    struct Matrix bias = nnet->bias[layer];

    matmul(sInterval->eq_matrix, &weights, sInterval->new_eq_matrix);
    if (err_row > 0)
    {
        (*sInterval->err_matrix).row = ERR_NODE;
        matmul(sInterval->err_matrix, &weights, sInterval->new_err_matrix);
        (*sInterval->new_err_matrix).row = (*sInterval->err_matrix).row = err_row;
    }

    for (int i = 0; i < layerSize; i++)
    {
        long long bias_index = inputSize + i * (inputSize + 1);
        (*sInterval->new_eq_matrix).data[bias_index] += bias.data[i];
    }

    (*sInterval->err_matrix).col = (*sInterval->new_err_matrix).col =
        nnet->layerSizes[layer + 1];
}

void sym_conv_layer(struct SymInterval *sInterval, struct NNet *nnet,
                    int layer, int err_row)
{
    // start handling conv layers
    int inputSize = nnet->inputSize;
    int layerSize = nnet->layerSizes[layer + 1];
    (*sInterval->new_eq_matrix).row = inputSize + 1;
    (*sInterval->new_eq_matrix).col = layerSize;
    (*sInterval->err_matrix).row = (*sInterval->new_err_matrix).row = err_row;
    (*sInterval->err_matrix).col = (*sInterval->new_err_matrix).col = layerSize;

    //layer is conv
    int out_channel = nnet->convLayer[layer][0];
    int in_channel = nnet->convLayer[layer][1];
    int kernel_size = nnet->convLayer[layer][2];
    int stride = nnet->convLayer[layer][3];
    int padding = nnet->convLayer[layer][4];
    int lt_pad = padding / 2;
    //size is the input size in each channel
    int size = sqrt(nnet->layerSizes[layer] / in_channel);
    //padding size is the input size after padding
    int padding_size = size + padding;
    //out_size is the output size in each channel after kernel
    int out_size = ceil((padding_size + 1.0 - kernel_size) / stride);

    float *new_new_equation = (float *)malloc(sizeof(float) *
                                              padding_size * padding_size * in_channel * (inputSize + 1));
    memset(new_new_equation, 0, sizeof(float) * padding_size * padding_size * in_channel * (inputSize + 1));
    float *new_new_equation_err = (float *)malloc(sizeof(float) *
                                                  padding_size * padding_size * in_channel * ERR_NODE);
    memset(new_new_equation_err, 0, sizeof(float) * padding_size * padding_size * in_channel * ERR_NODE);

    for (int ic = 0; ic < in_channel; ic++)
    {
        for (int h = 0; h < size; h++)
        {
            for (int w = 0; w < size; w++)
            {
                for (int k = 0; k < inputSize + 1; k++)
                {
                    long long loc_nn = (ic * padding_size * padding_size + padding_size * (h + lt_pad) + w + lt_pad) * (inputSize + 1) + k;
                    long long loc_eq = (ic * size * size + size * h + w) * (inputSize + 1) + k;
                    new_new_equation[loc_nn] = (*sInterval->eq_matrix).data[loc_eq];
                }
                for (int k = 0; k < err_row; k++)
                {
                    long long loc_nn = (ic * padding_size * padding_size + padding_size * (h + lt_pad) + w + lt_pad) * (ERR_NODE) + k;
                    long long loc_eq = (ic * size * size + size * h + w) * (ERR_NODE) + k;
                    new_new_equation_err[loc_nn] = (*sInterval->err_matrix).data[loc_eq];
                }
            }
        }
    }

    for (int oc = 0; oc < out_channel; oc++)
    {
        for (int oh = 0; oh < out_size; oh++)
        {
            for (int ow = 0; ow < out_size; ow++)
            {
                if (err_row > 0)
                {
                    int start = ow * stride + oh * stride * padding_size;
                    for (int k = 0; k < inputSize + 1; k++)
                    {
                        for (int kh = 0; kh < kernel_size; kh++)
                        {
                            for (int kw = 0; kw < kernel_size; kw++)
                            {
                                for (int ic = 0; ic < in_channel; ic++)
                                {
                                    long long loc_eq = (oc * out_size * out_size +
                                                        oh * out_size + ow) *
                                                           (inputSize + 1) +
                                                       k;
                                    long long loc_nn = (ic * padding_size * padding_size +
                                                        padding_size * kh + kw + start) *
                                                           (inputSize + 1) +
                                                       k;
                                    (*sInterval->new_eq_matrix).data[loc_eq] +=
                                        nnet->conv_matrix[layer][oc][ic][kh * kernel_size + kw] *
                                        new_new_equation[loc_nn];
                                }
                            }
                        }
                        if (k == inputSize)
                        {
                            (*sInterval->new_eq_matrix).data[(oc * out_size * out_size + ow + oh * out_size) * (inputSize + 1) + k] += nnet->conv_bias[layer][oc];
                        }
                    }
                    for (int k = 0; k < err_row; k++)
                    {
                        for (int kh = 0; kh < kernel_size; kh++)
                        {
                            for (int kw = 0; kw < kernel_size; kw++)
                            {
                                for (int ic = 0; ic < in_channel; ic++)
                                {
                                    long long loc_er = (oc * out_size * out_size + oh * out_size + ow) * ERR_NODE + k;
                                    long long loc_nn = (ic * padding_size * padding_size +
                                                        padding_size * kh + kw + start) *
                                                           ERR_NODE +
                                                       k;
                                    (*sInterval->new_err_matrix).data[loc_er] +=
                                        nnet->conv_matrix[layer][oc][ic][kh * kernel_size + kw] *
                                        new_new_equation_err[loc_nn];
                                }
                            }
                        }
                    }
                }
                else
                {
                    int start = ow * stride + oh * stride * padding_size;
                    for (int k = 0; k < inputSize + 1; k++)
                    {
                        for (int kh = 0; kh < kernel_size; kh++)
                        {
                            for (int kw = 0; kw < kernel_size; kw++)
                            {
                                for (int ic = 0; ic < in_channel; ic++)
                                {
                                    (*sInterval->new_eq_matrix).data[(oc * out_size * out_size + oh * out_size + ow) * (inputSize + 1) + k] += nnet->conv_matrix[layer][oc][ic][kh * kernel_size + kw] *
                                                                                                                                               new_new_equation[(ic * padding_size * padding_size + padding_size * kh + kw + start) * (inputSize + 1) + k];
                                }
                            }
                        }
                        if (k == inputSize)
                        {
                            long long loc_eq = (oc * out_size * out_size + ow + oh * out_size) *
                                                   (inputSize + 1) +
                                               k;
                            (*sInterval->new_eq_matrix).data[loc_eq] +=
                                nnet->conv_bias[layer][oc];
                        }
                    }
                }
            }
        }
    }
    free(new_new_equation);
    free(new_new_equation_err);
}

// calculate the upper and lower bound for the ith node in each layer
void relu_bound(struct SymInterval *sInterval, struct NNet *nnet,
                struct Interval *input, int i, int layer, int err_row,
                float *low, float *up)
{
    float tempVal_upper = 0.0, tempVal_lower = 0.0;
    int inputSize = nnet->inputSize;

    for (int k = 0; k < inputSize; k++)
    {
        if ((*sInterval->new_eq_matrix).data[k + i * (inputSize + 1)] >= 0)
        {
            tempVal_lower +=
                (*sInterval->new_eq_matrix).data[k + i * (inputSize + 1)] *
                input->lower_matrix->data[k];
            tempVal_upper +=
                (*sInterval->new_eq_matrix).data[k + i * (inputSize + 1)] *
                input->upper_matrix->data[k];
        }
        else
        {
            tempVal_lower +=
                (*sInterval->new_eq_matrix).data[k + i * (inputSize + 1)] *
                input->upper_matrix->data[k];
            tempVal_upper +=
                (*sInterval->new_eq_matrix).data[k + i * (inputSize + 1)] *
                input->lower_matrix->data[k];
        }
    }

    tempVal_lower += (*sInterval->new_eq_matrix).data[inputSize + i * (inputSize + 1)];
    tempVal_upper += (*sInterval->new_eq_matrix).data[inputSize + i * (inputSize + 1)];

    if (err_row > 0)
    {
        for (int err_ind = 0; err_ind < err_row; err_ind++)
        {
            if ((*sInterval->new_err_matrix).data[err_ind + i * ERR_NODE] > 0)
            {
                tempVal_upper += (*sInterval->new_err_matrix).data[err_ind + i * ERR_NODE];
            }
            else
            {
                tempVal_lower += (*sInterval->new_err_matrix).data[err_ind + i * ERR_NODE];
            }
        }
    }

    *up = tempVal_upper;
    *low = tempVal_lower;
}

// relax the relu layers and get the new symbolic equations
int sym_relu_layer(struct SymInterval *sInterval,
                   struct Interval *input,
                   struct Interval *output,
                   struct NNet *nnet,
                   int R[][nnet->maxLayerSize],
                   int layer,
                   int err_row,
                   int *wrong_nodes,
                   int *wrong_node_length,
                   int *node_cnt)
{
    float tempVal_upper = 0.0, tempVal_lower = 0.0;

    int inputSize = nnet->inputSize;
    int layerSize = nnet->layerSizes[layer + 1];

    //record the number of wrong nodes
    int wcnt = 0;
    for (int i = 0; i < layerSize; i++)
    {
        relu_bound(sInterval, nnet, input, i, layer, err_row,
                   &tempVal_lower, &tempVal_upper);

        //Perform ReLU relaxation
        if (layer < (nnet->numLayers - 1))
        {
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
                R[layer][i] = 0;
            }
            else if (tempVal_lower >= 0.0)
            {
                R[layer][i] = 2;
            }
            else
            {
                //wrong node length includes the wrong nodes in convolutional layers
                wrong_nodes[*wrong_node_length] = *node_cnt;

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

                // long long err_index = (long long)*wrong_node_length + ((long long)i) * ((long long)ERR_NODE);
                long long err_index = *wrong_node_length + i * ERR_NODE;
                (*sInterval->new_err_matrix)
                    .data[err_index] -=
                    tempVal_upper * tempVal_lower /
                    (tempVal_upper - tempVal_lower);

                *wrong_node_length += 1;
                wcnt += 1;

                R[layer][i] = 1;
            }
        }
        else
        {
            output->upper_matrix->data[i] = tempVal_upper;
            output->lower_matrix->data[i] = tempVal_lower;
        }
        *node_cnt += 1;
    }
    return wcnt;
}

int forward_prop_interval_equation_linear_conv(struct NNet *network,
                                               struct Interval *input,
                                               struct Interval *output,
                                               float *grad,
                                               struct SymInterval *sInterval,
                                               int *wrong_nodes,
                                               int *wrong_node_length)
{
    int node_cnt = 0;

    struct NNet *nnet = network;
    int numLayers = nnet->numLayers;
    int inputSize = nnet->inputSize;
    int maxLayerSize = nnet->maxLayerSize;

    int R[numLayers][maxLayerSize];
    memset(R, 0, sizeof(float) * numLayers * maxLayerSize);

    float *equation = sInterval->eq_matrix->data;
    float *new_equation = sInterval->new_eq_matrix->data;
    float *equation_err = sInterval->err_matrix->data;
    float *new_equation_err = sInterval->new_err_matrix->data;

    memset(equation, 0, sizeof(float) * (inputSize + 1) * maxLayerSize);
    memset(equation_err, 0, sizeof(float) * ERR_NODE * maxLayerSize);
    for (int i = 0; i < inputSize; i++)
    {
        equation[i * (inputSize + 1) + i] = 1;
    }

    //err_row is the number that is wrong before current layer
    int err_row = 0;
    for (int layer = 0; layer < (numLayers); layer++)
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
        sym_relu_layer(sInterval,
                       input,
                       output,
                       nnet,
                       R,
                       layer,
                       err_row,
                       wrong_nodes,
                       wrong_node_length,
                       &node_cnt);

        memcpy(equation, new_equation, sizeof(float) * (inputSize + 1) * maxLayerSize);
        memcpy(equation_err, new_equation_err, sizeof(float) * (ERR_NODE)*maxLayerSize);
        sInterval->eq_matrix->row = sInterval->new_eq_matrix->row;
        sInterval->eq_matrix->col = sInterval->new_eq_matrix->col;
        sInterval->err_matrix->row = sInterval->new_err_matrix->row;
        sInterval->err_matrix->col = sInterval->new_err_matrix->col;
        err_row = *wrong_node_length;
    }

    if (grad != NULL)
        backward_prop_conv(nnet, grad, R);

    return 1;
}
