//
//  techniques.cpp
//  Coordinate_descent
//
//  Created by Zhiwei Fan on 10/13/15.
//  Edited by Huawei Wang on 10/03/18, new CUDA 5.0 version
//  Copyright © 2015 Zhiwei Fan. All rights reserved.
//

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <time.h>
#include "techniques.h"
#include "DataManagement.h"
#include "linear_models.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include "math.h"
techniques::techniques(){};

/*
lm becomes class variable, calls device function like below
//Compute the partial gradient
// if to avoid overhead, may be here not much dimension of this single model, just update paramas in the host
    cudaMemcopy(dmodel, model + j, 1*sizeof(double), cudaMemcpyHostToDevice);
*/

__global__ void gradientkl(double* dY, double* dH, double* dX, double* dmul_arr, long row_num) {
    int Idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(Idx < row_num) {
        dmul_arr[Idx] =  -dY[Idx]/(1+exp(dY[Idx]*dH[Idx]))*dX[Idx];
    }
    
}

__global__ void backkl(double* dH, double* dX, double diff, long row_num) {
	int Idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(Idx < row_num) {
        // dH[Idx] = dH[Idx] + diff * dX[Idx];
        dH[Idx] = __fma_rn(diff, dX[Idx], dH[Idx]);
    }
}

__global__ void losskl(double* dY, double* dH, double* dFcur, long row_num) {
	int Idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(Idx < row_num) {
        // dFcur[Idx] = log1pf(-expf(dY[Idx]*dH[Idx]));
        dFcur[Idx] = log(1 + exp(-dY[Idx]*dH[Idx]));
    }
}
/**
 Coordinate Descent/Block Coordinate Descent:
 (CD/BCD)
 Materialize, Stream, Factorize
 
 Stochastic Gradient Descent/Batch Gradient Descent:
 (SGD/BGD)
 Materialize only

 **/
//Just Logistic Regression by now
__host__ void techniques::materialize(string table_T, setting _setting, double *model)
{
    DataManagement DM;
    DM.message("Start materialize");
    linear_models lm;
    vector<long> tableInfo(3);
    vector<string> fields = DM.getFieldNames(table_T, tableInfo);
    int feature_num = (int)tableInfo[1];
    long row_num = 100000; //tableInfo[2];
    
    double *Y;
    double *H;
    double *X;
    double *mul_arr;

    cudaEvent_t startEvent_exc, stopEvent_exc;
	cudaEventCreate(&startEvent_exc);
	cudaEventCreate(&stopEvent_exc);
	float elapsedTime_exc;

    //setting
    double step_size = _setting.step_size;
    
    //Allocate the memory to the model
    model = new double[feature_num];
    //Allocate the memory to the label Array
    Y = new double[row_num];
    //Allocate the memory to H
    H = new double[row_num];
    //Allocate the memory to X
    X = new double[row_num];
    // Temp variable for recording the multiply value
	mul_arr = new double[row_num];
    double *dY;
    double *dH;
    double *dX;
    double *dmul_arr;
    double *dFcur;
    // double *dmodel;
	// Allocate Device variables need for computing
    cudaMalloc((void**)&dY, row_num*sizeof(double));
    cudaMalloc((void**)&dH, row_num*sizeof(double));
    cudaMalloc((void**)&dX, row_num*sizeof(double)); 
    cudaMalloc((void**)&dmul_arr, row_num*sizeof(double));   //临时变量          
    // cudaMalloc((void**)&dmodel, 1*sizeof(double));           //计算变量                        
    cudaMalloc((void**)&dFcur, row_num*sizeof(double));
    cudaMemset(dH, 0, row_num*sizeof(double));
	// kernel parameter
    const int threadsPerBlock = 1024;
    const int blocksPerGrid = row_num/threadsPerBlock + 1;
    
    double F = 0.00;
    double F_partial = 0.00;
    double r_curr = 0.00;
    double r_prev = 0.00;
    int k = 0;
    
    
    for(int i = 0; i < feature_num; i ++)
    {
        model[i] = 0.00;
    }
   
	// Fetch the label column 1 time, leave it on GPU memory
	// May move to shared memory to improve efficiency
 
    DM.fetchColumn(fields[1], row_num, Y);
    cudaMemcpy(dY, Y, row_num*sizeof(double), cudaMemcpyHostToDevice);
    //First do Logistic Regression
    do
    {
        //Update one coordinate each time
        for(int j = 0; j < feature_num; j ++)
        {
            F_partial = 0.00;
            //Fetch the each column and store the current column into X
            DM.fetchColumn(fields[2+j], row_num, X);
	        
            cudaEventRecord(startEvent_exc,0); // staring timing for exclusive
            cudaMemcpy(dX, X, row_num*sizeof(double), cudaMemcpyHostToDevice);
            // launch the kernal only 1 time
            cudaEventRecord(stopEvent_exc,0);  // ending timing for exclusive
            cudaEventSynchronize(stopEvent_exc);   
            cudaEventElapsedTime(&elapsedTime_exc, startEvent_exc, stopEvent_exc);
            cout << "Gradient recuce time " << elapsedTime_exc << endl;

            gradientkl<<<blocksPerGrid, threadsPerBlock>>>(dY, dH, dX, dmul_arr, row_num);
			cudaDeviceSynchronize();

            // Reduce to get the sum of current gradient
			cudaMemcpy(mul_arr, dmul_arr, row_num*sizeof(double), cudaMemcpyDeviceToHost);
			F_partial = thrust::reduce(thrust::device, dmul_arr, dmul_arr + row_num, 0.0);

            // for(long i = 0; i < row_num; i++) {
            //     F_partial += mul_arr[i];
            // }

            double W_j = model[j];
            //Update the current coordinate
            model[j]  = model[j] - step_size * F_partial;
            double diff = model[j] - W_j;
            
            backkl<<<blocksPerGrid, threadsPerBlock>>>(dH, dX, diff, row_num);
            cudaDeviceSynchronize();
        }
        cudaMemcpy(H, dH, row_num*sizeof(double), cudaMemcpyDeviceToHost);										
        r_prev = F;
        //Caculate F
        F = 0.00;
        /*
        for(long i = 0; i < row_num ; i ++)
        {
            double tmp = lm.Fe_lr(Y[i],H[i]);
            
            if(i < 5){
                cout<<"Yi: "<<Y[i]<<","<<"Hi: "<<H[i]<<","<<"tmp: "<<tmp<<endl;
            }
            
            
            F += tmp;
        }
        */
        
        losskl<<<blocksPerGrid, threadsPerBlock>>>(dY, dH, dFcur, row_num);
        
        F = thrust::reduce(thrust::device, dFcur, dFcur + row_num, 0.0);
        r_curr = F;

        k++;
        cout<<"Model: "<<endl;
        
        for(int i = 0; i < 10; i ++)
        {
            cout<<model[i]<< " ";
        }
        
        //cout<<endl;
        cout<<"Loss value: "<<F<<endl;
        
    }
    while(!stop(k,r_prev,r_curr,_setting));
    cudaFree(dY);
    cudaFree(dH);
    cudaFree(dX);
    cudaFree(dmul_arr);
    // cudaFree(dmodel);
    delete [] Y;
    delete [] X;
    delete [] H;
    
    printf("The final loss: %lf\n",r_curr);
    printf("Number of iteration: %d\n",k);
    printf("Model: ");
    for(int i = 0; i < feature_num; i ++)
    {
        if(i == feature_num - 1)
        {
            printf("%.20f\n",model[i]);
        }
        else
        {
             printf("%.20f, ",model[i]);
        }
       
    }
    
    DM.message("Finish materialize");
    
}

/* Should be no oid-oid mapping here */
__host__ void techniques::stream(string table_S, string table_R, setting _setting, double *model)
{
    DataManagement DM;
    DM.message("Start stream");
    // linear_models lm; set as class variable
    
    //Get the table information and column names
    vector<long> tableInfo_S(3);
    vector<long> tableInfo_R(3);
    vector<string> fields_S = DM.getFieldNames(table_S, tableInfo_S);
    vector<string> fields_R = DM.getFieldNames(table_R, tableInfo_R);
    int feature_num_S = (int)tableInfo_S[1];
    int feature_num_R = (int)tableInfo_R[1];
    int feature_num = feature_num_S + feature_num_R;
    long row_num = tableInfo_S[2];
    long row_num_R = tableInfo_R[2];
    
    double *Y;
    double *H;
    double *X;
    
    //setting
    double step_size = _setting.step_size;
    
    //Allocate the memory to the model
    model = new double[feature_num_S + feature_num_R];
    //Allocate the memory to the label Array
    Y = new double[row_num];
    //Allocate the memory to H
    H = new double[row_num];
    //Allocate the memory to X
    X = new double[row_num];
    
    double F = 0.00;
    double F_partial = 0.00;
    double r_curr = 0.00;
    double r_prev = 0.00;
    int k = 0;
    
    for(int i = 0; i < feature_num; i ++)
    {
        model[i] = 0.00;
        
    }
    
    for(int i = 0; i < row_num; i ++)
    {
        H[i] = 0.00;
    }
    
    DM.fetchColumn(fields_S[1], row_num, Y);
    
    //First do Logistic Regression
    do
    {
        printf("Start fetching KKMR reference\n");
        //OID-OID Mapping (Key Foreign-Key Mapping Reference)
        double *KKMR = new double[row_num];
        //Read the fk column(referred rid in R) in table S, rid column in R
        ifstream fk;
        //Load the fk to KKMR
        fk.open(fields_S[2], ios::in | ios::binary);
        //rid.open(table2_fields[0], ios::in | ios::binary);
        if(!fk.is_open())
        {
            cerr<<"Error Message: "<<"Cannot load the fk column."<<endl;
            exit(1);
        }
        fk.read((char *)KKMR, row_num*(sizeof(double)));
        fk.close();
        printf("Finish fetchig KKMR reference\n");
        
        //Update one coordinate each time
        for(int j = 0; j < feature_num; j ++)
        {
            F_partial = 0.00;
            //Fetch each column in table S and store the current column into X
            if(j < feature_num_S)
            {
                DM.fetchColumn(fields_S[3+j], row_num, X);
            }
            else
            {
                double *tmp_buffer = new double[row_num_R];
                DM.fetchColumn(fields_R[1+j-feature_num_S], row_num_R, tmp_buffer);
                for(long m = 0; m < row_num; m ++)
                {
                    long fk = KKMR[m];
                    X[m]= tmp_buffer[fk-1];
                }
                delete tmp_buffer;
            }
            
//////////////////////These part the same as materialize before may just consider wrapping into a single function
            //Compute the partial gradient
            for(long i = 0; i < row_num ; i ++)
            {
                F_partial += lm.G_lr(Y[i],H[i])*X[i];
            }
            
            //Store the old W(j)
            double W_j = model[j];
            
            //Update the current coordinate
            model[j] = model[j] - step_size * F_partial;
            
            
            double diff = model[j] - W_j;
            
            //Update the intermediate variable
            //H = H + (Wj - old_Wj)* X(,j)
            for(long m = 0; m < row_num; m ++ )
            {
                H[m] = H[m] + diff*X[m];
            }
        }
        
        r_prev = F;
        //Caculate F
        F = 0.00;
        for(long i = 0; i < row_num ; i ++)
        {
            double tmp = lm.Fe_lr(Y[i],H[i]);
            F += tmp;
        }
        
        r_curr = F;
        k++;
        
        delete [] KKMR;
    } while(!stop(k,r_prev,r_curr,_setting));
    
    delete [] Y;
    delete [] X;
    delete [] H;
    
    
    printf("The final loss: %lf\n",r_curr);
    printf("Number of iteration: %d\n",k);
    printf("Model: ");
    for(int i = 0; i < feature_num; i ++)
    {
        if(i == feature_num - 1)
        {
            printf("%.20f\n",model[i]);
        }
        else
        {
            printf("%.20f, ",model[i]);
        }
        
    }
    
    DM.message("Finish stream");
}


__host__ void techniques::factorize(string table_S, string table_R, setting _setting, double *model)
{
    DataManagement DM;
    DM.message("Start factorize");
    linear_models lm;
    
    //Get the table information and column names
    vector<long> tableInfo_S(3);
    vector<long> tableInfo_R(3);
    vector<string> fields_S = DM.getFieldNames(table_S, tableInfo_S);
    vector<string> fields_R = DM.getFieldNames(table_R, tableInfo_R);
    int feature_num_S = (int)tableInfo_S[1];
    int feature_num_R = (int)tableInfo_R[1];
    int feature_num = feature_num_S + feature_num_R;
    long row_num = tableInfo_S[2];
    long row_num_R = tableInfo_R[2];
    
    printf("Start fetching KKMR reference\n");
    //OID-OID Mapping (Key Foreign-Key Mapping Reference)
    double *KKMR = new double[row_num];
    //Read the fk column(referred rid in R) in table S, rid column in R
    ifstream fk;
    //Load the fk to KKMR
    fk.open(fields_S[2], ios::in | ios::binary);
    //rid.open(table2_fields[0], ios::in | ios::binary);
    if(!fk.is_open())
    {
        cerr<<"Error Message: "<<"Cannot load the fk column."<<endl;
        exit(1);
    }
    fk.read((char *)KKMR, row_num*(sizeof(double)));
    fk.close();
    //printf("Finish fetchig KKMR reference\n");
    
    double *Y;
    double *H;
    double *X;
    
    //setting
    double step_size = _setting.step_size;
    
    //Allocate the memory to the model
    model = new double[feature_num_S + feature_num_R];
    //Allocate the memory to the label Array
    Y = new double[row_num];
    //Allocate the memory to H
    H = new double[row_num];
    //Allocate the memory to X
    X = new double[row_num];
    
    double F = 0.00;
    double F_partial = 0.00;
    double r_curr = 0.00;
    double r_prev = 0.00;
    int k = 0;
    
    for(int i = 0; i < feature_num; i ++)
    {
        model[i] = 0.00;
        
    }
    
    for(int i = 0; i < row_num; i ++)
    {
        H[i] = 0.00;
    }
    
    DM.fetchColumn(fields_S[1], row_num, Y);
    
    //First do Logistic Regression
    do
    {
        //Update one coordinate each time
        for(int j = 0; j < feature_num; j ++)
        {
            F_partial = 0.00;
            //Fetch each column in table S and store the current column into X
            if(j < feature_num_S)
            {
                DM.fetchColumn(fields_S[3+j], row_num, X);
//////////////////////////////////////////////////////Same part as before
                //Compute the partial gradient
                for(int i = 0; i < row_num ; i ++)
                {
                    F_partial += lm.G_lr(Y[i],H[i])*X[i];
                }
                
                //Store the old W(j)
                double W_j = model[j];
                
                //Update the current coordinate
                model[j] = model[j] - step_size * F_partial;
                cout<<"model("<<j<<"): "<<model[j]<<endl;
                
                double diff = model[j] - W_j;
                //Update the intermediate variable
                //H = H + (Wj - old_Wj)* X(,j)
                for(long m = 0; m < row_num; m ++ )
                {
                    H[m] = H[m] + diff*X[m];
                }
            }
            else
            {
                double **tmp_buffer = new double*[2];
                for(int i = 0; i < 2; i ++)
                {
                    tmp_buffer[i] = new double[row_num_R];
                    for(long k = 0; k < row_num_R; k ++)
                    {
                        tmp_buffer[i][k] = 0.00;
                    }
                }
                
                DM.fetchColumn(fields_R[1+j-feature_num_S], row_num_R, tmp_buffer[0]);
                //Compute the partial gradient
                for(long m = 0; m < row_num; m ++)
                {
                    long fk = KKMR[m];
                    tmp_buffer[1][fk-1] += lm.G_lr(Y[m],H[m]);
                }
                for(long j = 0; j < row_num_R; j ++)
                {
                    F_partial += tmp_buffer[0][j]*tmp_buffer[1][j];
                }
                
                
                //Store the old W(j)
                double W_j = model[j];
                
                //Update the current coordinate
                model[j] = model[j] - step_size * F_partial;
                cout<<"model("<<j<<"): "<<model[j]<<endl;
                
                double diff = model[j] - W_j;

                
                //Factorized computation
                for(long k = 0; k < row_num_R; k ++)
                {
                    tmp_buffer[0][k] = diff*tmp_buffer[0][k];
                }
                
                //Update the intermediate variable
                //H = H + (Wj - old_Wj)* X(,j)
                for(long m = 0; m < row_num; m ++ )
                {
                    long fk = KKMR[m];
                    H[m] = H[m] + tmp_buffer[0][fk-1];
                }

                
                delete [] tmp_buffer[0];
                delete [] tmp_buffer[1];
                delete [] tmp_buffer;
            }
            
        }
        
        r_prev = F;
        //Caculate F
        F = 0.00;
        for(int i = 0; i < row_num ; i ++)
        {
            double tmp = lm.Fe_lr(Y[i],H[i]);
            F += tmp;
        }
        
        r_curr = F;
        k++;
    }
    while(!stop(k,r_prev,r_curr,_setting));
    
    delete [] Y;
    delete [] X;
    delete [] H;
    delete [] KKMR;
    
    printf("The final loss: %lf\n",r_curr);
    printf("Number of iteration: %d\n",k);
    printf("Model: ");
    for(int i = 0; i < feature_num; i ++)
    {
        if(i == feature_num - 1)
        {
            printf("%.20f\n",model[i]);
        }
        else
        {
            printf("%.20f, ",model[i]);
        }
        
    }
    
    DM.message("Finish factorize");

}

__host__ bool techniques::stop(int k, double r_prev, double r_curr, setting &setting)
{
    double iter_num = k;
    double difference = abs(r_prev - r_curr);
    
    if( iter_num == setting.iter_num || difference <= setting.error)
    {
        return true;
    }
    else
    {
        return false;
    }
}


/*
 Read a single file the columns of which are in format like: id, label, feature
 The offset entry for W0 is not considered for now
 Logistic Regression for now
 */

//specific techniques selection: flag (for generalization purpose)

__host__ void techniques::SGD(vector< vector<double> > data, setting _setting, double *&model, int feature_num)
{
    DataManagement::message("Start SGD");
    long data_size = data.size();
    vector<long> original_index_set;
    vector<long> shuffling_index;
    //Initialize the original_index_set
    for(long i = 0; i < data_size; i ++)
    {
        original_index_set.push_back(i);
    }
    
    linear_models lm;
    //setting
    double step_size = _setting.step_size;
    
    //Allocate the memory to model
    model = new double[feature_num];
    
    for(int i = 0; i < feature_num; i ++)
    {
        model[i] = 0.00;
        
    }
    
    //Loss Function
    double F = 0.00;
    double r_curr = 0.00;
    double r_prev = 0.00;
    int k = 0;
    
    do
    {
        r_prev = F;
        F = 0.00;
        vector<double> gradient(feature_num,0.00);
        
        //Shuffling
        shuffling_index = shuffle(original_index_set, (unsigned)time(NULL));
        /**
        cout<<"shuffling order:"<<endl;
        for(int i = 0; i < shuffling_index.size(); i ++)
        {
            cout<<shuffling_index.at(i)<<" ";
        }
        cout<<endl;
        **/
        
        for(long j = 0; j < data_size; j ++)
        {
            long cur_index = shuffling_index[j];
            
            //Update the model
            double output = 0.00;
            for(int k = 0; k < feature_num; k ++)
            {
                output += model[k]*data[cur_index][k+2];
            }
            
            for(int k = 0; k < feature_num; k ++)
            {
                gradient[k] = lm.G_lr(data[cur_index][1],output)*data[cur_index][2+k];
                model[k] = model[k]-step_size*gradient[k];
            }
            
        }
        
        //Calculate F
        for(long j = 0; j < data_size; j ++)
        {
            double output = 0.00;
            for(int k = 0; k < feature_num; k ++)
            {
                output += model[k]*data[j][k+2];
            }
            double tmp = lm.Fe_lr(data[j][1], output);
            F += tmp;
        }
        
        r_curr = F;
        k ++;
    }
    while(!stop(k,r_prev,r_curr,_setting));
    
    printf("The final loss: %lf\n",r_curr);
    printf("Number of iteration: %d\n",k);
    printf("Model: ");
    for(int i = 0; i < feature_num; i ++)
    {
        if(i == feature_num - 1)
        {
            printf("%.20f\n",model[i]);
        }
        else
        {
            printf("%.20f, ",model[i]);
        }
        
    }
    
    DataManagement::message("Finish SGD");

}

__host__  void techniques::BGD(vector< vector<double> > data, setting _setting, double *&model, int feature_num)
{
    DataManagement::message("Start BGD");
    long data_size = data.size();
    
    // linear_models lm;
    //setting
    double step_size = _setting.step_size;
    
    //Allocate the memory to the model
    model = new double[feature_num];
    
    for(int i = 0; i < feature_num; i ++)
    {
        model[i] = 0.00;
    }
    
    //Loss Function
    double F = 0.00;
    double r_curr = 0.00;
    double r_prev = 0.00;
    int k = 0;
    
    do
    {
        r_prev = F;
        F = 0.00;
        vector<double> gradient(feature_num,0.00);
        
        for(long j = 0; j < data_size; j ++)
        {
            
            //Update the model
            double output = 0.00;
            for(int k = 0; k < feature_num; k ++)
            {
                output += model[k]*data[j][2+k];
            }
            
            for(int k = 0; k < feature_num; k ++)
            {
                gradient[k] += lm.G_lr(data[j][1],output)*data[j][2+k];
            }
            
        }
        
        
        for(int k = 0; k < feature_num; k ++)
        {
            model[k] = model[k]-step_size*gradient[k];
        }
        
        for(long j = 0; j < data_size; j ++)
        {
            double output = 0.00;
            for(int k = 0; k < feature_num; k ++)
            {
                output += model[k]*data[j][2+k];
            }
            double tmp = lm.Fe_lr(data[j][1], output);
            cout<<"tmp loss: "<<tmp<<endl;
            F += tmp;
        }
        
        
        r_curr = F;
        printf("The loss: %lf\n",F);
        k ++;
    }
    while(!stop(k,r_prev,r_curr,_setting));
    
    printf("The final loss: %lf\n",r_curr);
    printf("Number of iteration: %d\n",k);
    printf("Model: ");
    for(int i = 0; i < feature_num; i ++)
    {
        if(i == feature_num - 1)
        {
            printf("%.20f\n",model[i]);
        }
        else
        {
            printf("%.20f, ",model[i]);
        }
        
    }
    
    DataManagement::message("Finish BGD");

}

__host__ void techniques::classify(vector< vector<double> > data, vector<double> model)
{
    linear_models lm;
    // Count the number of correct classifcation
    long count = 0;
    long data_size =  data.size();
    if(data.at(0).size() != model.size()+2)
    {
        DataManagement::errorMessage("Inconsistent file provided");
    }
    
    int featureNum = (int)model.size();
    for(long i = 0; i < data_size; i ++)
    {
        double actual_label = data[i][1];
        double predicted_label = 0.00;
        double confidence = 0.00;
        double output = 0.00;
        for(int j = 0; j < featureNum; j ++)
        {
            output += model[j]*data[i][2+j];
        }
        cout<<"W^TX: "<<output<<endl;
        confidence = lm.C_lr(output);
        if(confidence > 0.5)
        {
            predicted_label = 1.00;
        }
        else
        {
            predicted_label = -1.00;
        }
        if(actual_label == predicted_label)
        {
            cout<<"Prediction Correct"<<endl;
            count++;
        }
        else
        {
            cout<<"Prediction Wrong"<<endl;
        }
        cout<<"Confidence: "<<confidence<<endl;
        cout<<"Actual Label: "<<actual_label<<","<<"Predicted Label: "<<predicted_label<<endl;
    }
    cout<<"Correcteness: "<<(double)count/(double)data_size<<endl;
    
    cout<<"Predict the newest day: "<<endl;
    vector<double> toBePredicted;
    toBePredicted.push_back(1);
    for(int i = 0; i < 4; i ++)
    {
        toBePredicted.push_back(0);
    }
    
    toBePredicted.push_back(7.93);
    toBePredicted.push_back(8.16);
    toBePredicted.push_back(7.87);
    toBePredicted.push_back(8.06);
    toBePredicted.push_back(23.3105);
    toBePredicted.push_back(8.06);
    toBePredicted.push_back(1);
    toBePredicted.push_back(7.93);
    toBePredicted.push_back(1);
    toBePredicted.push_back(0.15);
    
    double output = 0.00;
    for(int j = 0; j < featureNum; j ++)
    {
        output += model[j]*toBePredicted[j];
    }
    
}

vector<int> techniques::shuffle(vector<int> &index_set, unsigned seed)
{
    vector<int> original_set = index_set;
    int size = (int)index_set.size();
    vector<int> new_index_set;
    srand (seed);
    for(int i = 0; i < size; i ++)
    {
        int cur_size = (int)original_set.size();
        int rand_index = random()%cur_size;
        new_index_set.push_back(original_set.at(rand_index));
        original_set.erase(original_set.begin()+rand_index);
    }
    
    return new_index_set;
}

vector<long> techniques::shuffle(vector<long> &index_set, unsigned seed)
{
    vector<long> original_set = index_set;
    long size = (long)index_set.size();
    vector<long> new_index_set;
    srand(seed);
    for(long i = 0; i < size; i ++)
    {
        long cur_size = original_set.size();
        long rand_index = random()%cur_size;
        new_index_set.push_back(original_set.at(rand_index));
        original_set.erase(original_set.begin()+rand_index);
    }
    
    return new_index_set;
}

// Shuffle mainly focus on file operation, shuffle the index, not much computation, 
// No need for GPU launching
