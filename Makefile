all: DB

DB: Learning.o DataManagement.o techniquescuda.o
	nvcc -std=c++11  -O3 --use_fast_math -gencode arch=compute_61,code=sm_61 -I./ Learning.o DataManagement.o techniquescuda.o -o DB

Learning.o: Learning.cpp DataManagement.h techniques.h
	nvcc -std=c++11 -c -g -G  -O3 --use_fast_math -gencode arch=compute_61,code=sm_61 -I./ Learning.cpp

DataManagement.o: DataManagement.cpp DataManagement.h
	nvcc -std=c++11 -c -g -G  -O3 --use_fast_math -gencode arch=compute_61,code=sm_61 -I./ DataManagement.cpp

techniquescuda.o: techniquescuda.cu techniques.h DataManagement.h linear_models.h gradientkl.cu
	nvcc -std=c++11 -c -g -G  -O3 --use_fast_math -gencode arch=compute_61,code=sm_61 -I./ techniquescuda.cu gradientkl.cu

clean:
	rm -f *.o DB *~ #*
