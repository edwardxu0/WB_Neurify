#ifndef _NNETH_
#define _NNETH_

#include <math.h>
#include <string.h>
#include <time.h>

#include "lp_dev/lp_lib.h"
#include "matrix.h"
#include "interval.h"

extern long long ERR_NODE;
extern struct timeval start_time, finish_time;

//Neural Network Struct
struct NNet
{
    int symmetric;    //1 if network is symmetric, 0 otherwise
    int numLayers;    //Number of layers in the network
    int inputSize;    //Number of inputs to the network
    int outputSize;   //Number of outputs to the network
    int maxLayerSize; //Maximum size dimension of a layer in the network
    int *layerSizes;  //Array of the dimensions of the layers in the network
    int *layerTypes;  //Intermediate layer types

    /*
     * convlayersnum is the number of convolutional layers
     * convlayer is a matrix [convlayersnum][5]
     * out_channel, in_channel, kernel, stride, padding
    */
    int convLayersNum;
    int **convLayer;
    float ****conv_matrix;
    float **conv_bias;

    float ****matrix; //4D jagged array that stores the weights and biases
                      //the neural network.
    struct Matrix *weights;
    struct Matrix *bias;

    int target;
    struct Interval *output_constraint;
};

struct SymInterval
{
    struct Matrix *eq_matrix;
    struct Matrix *new_eq_matrix;
    struct Matrix *err_matrix;
    struct Matrix *new_err_matrix;
};

struct SymInterval *SymInterval_new(int inputSize, int maxLayerSize, int maxErrNodes);
void destroy_SymInterval(struct SymInterval *sym_interval);

void sym_fc_layer(struct SymInterval *sInterval, struct NNet *nnet, int layer, int err_row);

void sym_conv_layer(struct SymInterval *sInterval, struct NNet *nnet, int layer, int err_row);

void relu_bound(struct SymInterval *sInterval, struct NNet *nnet,
                struct Interval *input, int i, int layer, int err_row,
                float *low, float *up);

int sym_relu_layer(struct SymInterval *sInterval, struct Interval *input, struct Interval *output,
                   struct NNet *nnet, int R[][nnet->maxLayerSize],
                   int layer, int err_row,
                   int *wrong_nodes, int *wrong_node_length, int *node_cnt);

struct NNet *load_conv_network(const char *filename);
void destroy_conv_network(struct NNet *network);

void load_inputs(const char *input_path, int inputSize, float *input);

void initialize_input_interval(struct Interval *input_interval,
                               int inputSize,
                               float input[inputSize],
                               float epsilon);
void initialize_output_constraint(const char *path,
                                  struct Interval *output_interval,
                                  int outputSize);

void sort(float *array, int num, int *ind);

void sort_layers(int numLayers, int *layerSizes, int wrong_node_length, int *wrong_nodes);

void set_input_constraints(struct Interval *input, lprec *lp, int inputSize);

void set_node_constraints(lprec *lp, float *equation, int start, int sig, int inputSize);

float set_output_constraints(lprec *lp, float *equation, int start, int inputSize, int is_max, float *output, float *input_prev);

int evaluate_conv(struct NNet *network, struct Matrix *input, struct Matrix *output);

void backward_prop_conv(struct NNet *nnet, float *grad, int R[][nnet->maxLayerSize]);

int forward_prop_interval_equation_linear_conv(struct NNet *network,
                                               struct Interval *input,
                                               struct Interval *output,
                                               float *grad,
                                               int *wrong_nodes,
                                               int *wrong_node_length);

#endif