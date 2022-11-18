all: opencl1

opencl1: main.cpp makefile
	clang++ -framework OpenCL main.cpp -o opencl1
	
run: opencl1
	./opencl1
