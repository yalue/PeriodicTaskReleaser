all: launcher

launcher: launcher.o  runner.o util.o mm.o va.o
	nvcc launcher.o mm.o va.o runner.o util.o -o launcher -lpthread --cudart shared -g

runner.o: runner.c
	gcc -c runner.c -Wall -g

launcher.o: launcher.c
	gcc -c launcher.c -Wall -g

util.o: util.c
	gcc -c util.c -Wall -g

va.o: Samples/va.cu
	nvcc -c Samples/va.cu --cudart shared -g

mm.o: Samples/mm.cu
	nvcc -c Samples/mm.cu --cudart shared -g

clean:
	rm -f *.o launcher Samples/*.o
