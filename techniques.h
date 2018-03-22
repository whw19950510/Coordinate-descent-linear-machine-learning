//
//  techniques.hpp
//  Coordinate_descent
//
//  Created by Zhiwei Fan on 10/13/15.
//  Edited by Huawei Wang on 10/03/18, new CUDA 5.0 version
//  Copyright © 2015 Zhiwei Fan. All rights reserved.
//

#ifndef _techniques_
#define _techniques_

using namespace std;
#include "linear_models.h"
#include <stdio.h>
#include <string>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
struct setting
{
    int iter_num;
    double error;
    double step_size;
};

class techniques
{
public:
    linear_models lm;
    techniques();
    __host__ void materialize(string table_T, setting _setting, double *model);
    __host__ void stream(string table_S, string table_R, setting _setting, double *model);
    __host__ void factorize(string table_S, string table_R, setting _setting, double *model);
    __host__ bool stop(int k, double r_prev, double r_curr, setting &setting);
    __host__ void SGD(vector< vector<double> > data, setting _setting, double *&model, int feature_num);
    __host__ void BGD(vector< vector<double> > data, setting _setting, double *&model, int feature_num);
    __host__ void classify(vector< vector<double> > data, vector<double> model);
private:
    __host__ vector<int> shuffle(vector<int> &index_set, unsigned seed);
    __host__ vector<long> shuffle(vector<long> &index_set, unsigned seed);
};


#endif /* defined(_techniques_) */
