//
//  linear_models.cpp
//  Coordinate_descent
//
//  Created by Zhiwei Fan on 10/14/15.
//  Edited by Huawei Wang on 10/03/18, new CUDA 5.0 version
//  Copyright © 2015 Zhiwei Fan. All rights reserved.
//

#include "linear_models.h"
#include <cmath>

#define MAX(x,y) (x > y ? x : y)

linear_models::linear_models(){};

double linear_models::Fe_lr(double a, double b)
{
    double power = -(a*b);
    return log(1+pow(exp(1.00), power));
}

double linear_models::Fe_lsr(double a, double b)
{
    double base = a - b;
    return pow(base, 2);
}

double linear_models::Fe_lsvm(double a, double b)
{
    return MAX(0, 1 - a*b);
}

double linear_models::G_lr(double a, double b)
{
    double power = a*b;
    return -(a/(1+pow(exp(1.0),power)));
}

double linear_models::G_lsr(double a, double b)
{
    return 2*(b-a);
}

double linear_models::G_svm(double a, double b)
{
    if(a*b > 1)
    {
        return -1;
    }
    else
    {
        return 0;
    }
    return 0;
}

double linear_models::C_lr(double a)
{
    return 1.00/(double)(1+pow(exp(1.00),-a));
}
