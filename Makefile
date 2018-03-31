all: DB
# Warnings
WFLAGS	:= -Wall -Wextra -Wsign-conversion -Wsign-compare

# LIB & LINK SETTING
LIB := -L$(CUDA_HOME)/lib64 -lcudart
FLAGS := --ptxas-options=-v --use_fast_math

# Optimization and architecture, O3 optimization level for CPU code
OPT		:= -arch=sm_20 -O3
ARCH   	:= -march=native
# Language standard
CXXSTD	:= -std=c++11

# Linker options
LDOPT 	:= $(OPT)
LDFLAGS := 

DB: Learning.o DataManagement.o techniques.o linear_models.o
	nvcc -std=c++11  -O3 --use_fast_math -gencode arch=compute_61,code=sm_61 Learning.o DataManagement.o techniques.o linear_models.o -o DB	

Learning.o: Learning.cu DataManagement.h techniques.h
	nvcc -std=c++11 -c -g -G  -O3 --use_fast_math -gencode arch=compute_61,code=sm_61 Learning.cu
	
DataManagement.o: DataManagement.cpp DataManagement.h
	nvcc -std=c++11 -c -g -G  -O3 --use_fast_math -gencode arch=compute_61,code=sm_61 DataManagement.cpp

techniques.o: techniques.cu techniques.h DataManagement.h linear_models.h 
	nvcc -std=c++11 -c -g -G  -O3 --use_fast_math -gencode arch=compute_61,code=sm_61 techniques.cu	

linear_models.o: linear_models.cu linear_models.h
	nvcc -std=c++11 -c -g -G  -O3 --use_fast_math -gencode arch=compute_61,code=sm_61 linear_models.cu


clean:
	rm -f *.o coordinate_descent *~ #*
