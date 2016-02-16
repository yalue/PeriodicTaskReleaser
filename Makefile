all: launcher

launcher: launcher.o releaser.o runner.o util.o GPUOp.o
	nvcc launcher.o releaser.o GPUOp.o runner.o util.o -o launcher -lpthread --cudart shared -g

runner.o: runner.c
	gcc -c runner.c -Wall -g

releaser.o: releaser.c
	gcc -c releaser.c -Wall -g

launcher.o: launcher.c
	gcc -c launcher.c -Wall -g

util.o: util.c
	gcc -c util.c -Wall -g

GPUOp.o: GPUOp.cu
	nvcc -c GPUOp.cu --cudart shared -g

clean:
	rm -f *.o launcher
