
all:
	nvcc -c -arch=sm_61 kernel.cu 
	g++ -c main.cpp -I/usr/local/cuda/include
	g++ kernel.o main.o -L/usr/local/cuda/lib64 -lcudart -o test_exec	
