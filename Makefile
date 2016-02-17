all: launcher

launcher: launcher.o  runner.o util.o GPUOp.o
	nvcc launcher.o GPUOp.o runner.o util.o -o launcher -lpthread --cudart shared -g

runner.o: runner.c
	gcc -c runner.c -Wall -g

launcher.o: launcher.c
	gcc -c launcher.c -Wall -g

util.o: util.c
	gcc -c util.c -Wall -g

GPUOp.o: GPUOp.cu
	nvcc -c GPUOp.cu --cudart shared -g

clean:
	rm -f *.o launcher
