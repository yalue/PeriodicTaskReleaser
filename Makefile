all: launcher_va_c launcher_va_zc launcher_mm_c launcher_mm_zc

launcher_mm_zc: launcher.o  runner.o util.o mm_zc.o
	nvcc launcher.o mm_zc.o runner.o util.o --cudart shared -g -o launcher_mm_zc -lpthread

launcher_mm_c: launcher.o  runner.o util.o mm_c.o
	nvcc launcher.o mm_c.o runner.o util.o --cudart shared -g -o launcher_mm_c -lpthread

launcher_va_zc: launcher.o  runner.o util.o va_zc.o
	nvcc launcher.o va_zc.o runner.o util.o --cudart shared -g -o launcher_va_zc -lpthread

launcher_va_c: launcher.o  runner.o util.o va_c.o
	nvcc launcher.o va_c.o runner.o util.o --cudart shared -g -o launcher_va_c -lpthread

runner.o: runner.c
	gcc -c runner.c -Wall -g

launcher.o: launcher.c
	gcc -c launcher.c -Wall -g

util.o: util/util.c
	gcc -c util/util.c -Wall -g

mem_c.o: Samples/Copy/mem.cu
	nvcc -c Samples/Copy/mem.cu -o mem_c.o --cudart shared -g --ptxas-options=-v

va_c.o: Samples/Copy/va.cu
	nvcc -c Samples/Copy/va.cu -o va_c.o --cudart shared -g --ptxas-options=-v

mm_c.o: Samples/Copy/mm.cu
	nvcc -c Samples/Copy/mm.cu -o mm_c.o --cudart shared -g --ptxas-options=-v

mem_zc.o: Samples/ZeroCopy/mem.cu
	nvcc -c Samples/ZeroCopy/mem.cu -o mem_zc.o --cudart shared -g --ptxas-options=-v

va_zc.o: Samples/ZeroCopy/va.cu
	nvcc -c Samples/ZeroCopy/va.cu -o va_zc.o --cudart shared -g --ptxas-options=-v

mm_zc.o: Samples/ZeroCopy/mm.cu
	nvcc -c Samples/ZeroCopy/mm.cu -o mm_zc.o --cudart shared -g --ptxas-options=-v

clean:
	rm -rf *.o launcher_va_c launcher_va_zc launcher_mm_c launcher_mm_zc bin
