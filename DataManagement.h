//
//  DataManagement.h
//  Coordinate_descent
//
//  Created by Zhiwei Fan on 10/2/15.
//  Edited by Huawei Wang on 10/03/18, new CUDA 5.0 version
//  Copyright (c) 2015 Zhiwei Fan. All rights reserved.
//
// Store&getFieldNames include fileIO,join calls getFiledName funstion
#ifndef __Coordinate_descent__DataManagement__
#define __Coordinate_descent__DataManagement__

using namespace std;
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <string>
//Read data from table stored in text file and loaded into the DB
// RowStore is the only function called outside, to acquire data from outside world
// read data into meomory,这里应该加速不了
class DataManagement
{
public:
    DataManagement();
    
    void store(string FileName, int feature_num, int table_type, long row_num);
    void readColumn(string fileName, long row_num);
    void fetchColumn(string fileName, long row_num, double *col);
    void join(string table_name1, string table_name2, string joinTable);
    vector<string> getFieldNames(string tableName,vector<long> &tableInfo);
    void readTable(string tableName);
    static void message(string str);
    static void errorMessage(string str);
    vector< vector<double> > rowStore(string fileName);
private:
    vector<string> split(const string &s, char delim);
};

#endif /* defined(__Coordinate_descent__DataManagement__) */
