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
#ifndef _SPLITH_
#define _SPLITH_

#include <math.h>
#include <pthread.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include "nnet.h"

extern int MAX_DEPTH;
extern int MAX_THREAD;
extern int NEED_PRINT;

extern int adv_found;
extern int max_depth_exceeded;

#define PROGRESS_DEPTH 12
extern int progress_list[PROGRESS_DEPTH];
extern int total_progress[PROGRESS_DEPTH];

extern int count;

struct direct_run_check_conv_lp_args
{
	struct NNet *nnet;
	struct Interval *input;
	int *wrong_nodes;
	int wrong_node_length;
	int *sigs;
	int target;
	int sig;
	lprec *lp;
	int depth;
};

int check(struct NNet *nnet, struct Interval *output);
int check1(struct NNet *nnet, struct Matrix *output);
void check_adv1(struct NNet *nnet, struct Matrix *adv);

int forward_prop_interval_equation_conv_lp(struct NNet *nnet,
										   struct Interval *input,
										   int *sigs,
										   int target,
										   int sig,
										   lprec *lp);

int direct_run_check_conv_lp(struct NNet *nnet,
							 struct Interval *input,
							 int *wrong_nodes,
							 int wrong_node_length,
							 int *sigs,
							 int target,
							 int sig,
							 lprec *lp,
							 int depth);

int split_interval_conv_lp(struct NNet *nnet,
						   struct Interval *input,
						   int *wrong_nodes,
						   int wrong_node_length,
						   int *sigs,
						   lprec *lp,
						   int depth);

#endif