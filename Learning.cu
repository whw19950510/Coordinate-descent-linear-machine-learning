//
//  main.cpp
//  Coordinate_descent
//
//  Created by Zhiwei Fan on 10/2/15.
//  Edited by Huawei Wang on 10/03/18, new CUDA 5.0 version
//  Copyright (c) 2015 Zhiwei Fan. All rights reserved.
//

#include <iostream>
#include "DataManagement.h"
#include "techniques.h"
// Main function is here needs to define a kernal here, inorder to call
// the device function from the class
// defines a kernel according to differene needs, and prepare the data needed
// here, then calls the device function, finlly calls kernal in the main function

/*
 options:
 1. create table: DB create file_name(table_name) type feature_num row_num
 2. join table: DB join table_name1 table_name2 joinTable_name
 3. read table: DB read table_name
 4. materialize: DB m table_name
 5. stream: DB s table_name_S table_name_R
 6. factorize: DB f table_name_S table_name_R
 7. readColumn: DB column_name row_num
 8. Stochastic Gradient Descent: DB SGD train_file_name test_file_name
 9. Batch Gradient Descent:DB BGD train_file_name test_file_name
 */
int main(int argc, const char * argv[]) {
    
    
    DataManagement dM;
    techniques t;
    string option1 = "create";
    string option2 = "join";
    string option3 = "read";
    string option4 = "m";
    string option5 = "s";
    string option6 = "f";
    string option8 = "SGD";
    string option9 = "BGD";
    
    if(argc == 6 && argv[1] == option1)
    {
        //option 1
        string fileName = argv[2];
        int tableType = atoi(argv[3]);
        int featureNum = atoi(argv[4]);
        long rowNum = atol(argv[5]);
        dM.store(fileName, featureNum, tableType, rowNum);
    }
    else if(argc == 5 && argv[1] == option2)
    {
        //option 2
        string tableName1 = argv[2];
        string tableName2 = argv[3];
        string joinTableName = argv[4];
        dM.join(tableName1, tableName2, joinTableName);
    }
    else if(argc == 3 && argv[1] == option4)
    {
        string option = argv[1];
        string tableName = argv[2];
       
        
        double *model;
        int feature_num = 0;
        vector<double> model_vector;
        setting _setting;
        printf("Setting stepSize: \n");
        scanf("%lf",&_setting.step_size);
        //_setting.step_size = 0.01;
        printf("Setting error tolearence: \n");
        //_setting.error = 0.00005;
        scanf("%lf",&_setting.error);
        printf("Setting number of iterations: \n");
        _setting.iter_num = 10;
        scanf("%d",&_setting.iter_num);
        // 10 轮就收敛。。。这么夸张
        
        //Feature Number to be dealt with it later
        t.materialize(tableName, _setting, model);
        
        //delete[] model;
            
    }
    else if(argc == 4)
    {
        string tableName_S = argv[2];
        string tableName_R = argv[3];
        double *model;
        setting _setting;
        printf("Setting stepSize: \n");
        scanf("%lf",&_setting.step_size);
        //_setting.step_size = 0.01;
        printf("Setting error tolearence: \n");
        //_setting.error = 0.00005;
        scanf("%lf",&_setting.error);
        printf("Setting number of iterations: \n");
        scanf("%d",&_setting.iter_num);

        if(argv[1] == option5)
        {
            t.stream(tableName_S, tableName_R, _setting, model);
        }
        
        if(argv[1] == option6)
        {
            t.factorize(tableName_S, tableName_R, _setting, model);
        }
        
        //Print the model
        delete[] model;
    }
    else
    {
        cerr<<"Invalid command: wrong number of arguments or invalid option"<<endl;
        exit(1);
    }
    
    //dM.readTable("/Users/Hacker/Box Sync/Research/Coordinate_descent/Coordinate_descent/S");
    //dM.readTable("/Users/Hacker/Box Sync/Research/Coordinate_descent/Coordinate_descent/R");
  
    return 0;
}

