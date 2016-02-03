all: launcher

launcher: launcher.o releaser.o runner.o util.o GPUOp.o
	nvcc launcher.o releaser.o runner.o util.o GPUOp.o -o launcher -lpthread -Wall -O3

runner.o: runner.c
	gcc -c runner.c -Wall

releaser.o: releaser.c
	gcc -c releaser.c -Wall

launcher.o: launcher.c
	gcc -c launcher.c -Wall

util.o: util.c
	gcc -c util.c -Wall

GPUOp.o: GPUOp.cu
	nvcc -c GPUOp.cu 

clean:
	rm -f *.o launcher
