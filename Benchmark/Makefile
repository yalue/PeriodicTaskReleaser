SAMPLE_DIR = ../Samples
UTIL_DIR = ../util
INC = -I$(UTIL_DIR) -I$(SAMPLE_DIR)

all: benchmark_va_c benchmark_va_zc benchmark_mm_c benchmark_mm_zc benchmark_mem_c benchmark_mem_zc

benchmark_va_c: benchmark.o va_c.o util.o
	nvcc benchmark.o va_c.o util.o -o benchmark_va_c --cudart shared -g

benchmark_va_zc: benchmark.o va_zc.o util.o
	nvcc benchmark.o va_zc.o util.o -o benchmark_va_zc --cudart shared -g

benchmark_mm_c: benchmark.o mm_c.o util.o
	nvcc benchmark.o mm_c.o util.o -o benchmark_mm_c --cudart shared -g

benchmark_mm_zc: benchmark.o mm_zc.o util.o
	nvcc benchmark.o mm_zc.o util.o -o benchmark_mm_zc --cudart shared -g

benchmark_mem_c: benchmark.o mem_c.o util.o
	nvcc benchmark.o mem_c.o util.o -o benchmark_mem_c --cudart shared -g

benchmark_mem_zc: benchmark.o mem_zc.o util.o
	nvcc benchmark.o mem_zc.o util.o -o benchmark_mem_zc --cudart shared -g

benchmark.o: ./benchmark.c
	gcc -c ./benchmark.c -Wall -g $(INC)

memtest.o: ./memtest.c 
	gcc -c ./memtest.c -Wall -g

util.o: ../util/util.c
	gcc -c ../util/util.c -Wall -g

mem_c.o: ../Samples/Copy/mem.cu
	nvcc -c ../Samples/Copy/mem.cu -o mem_c.o --cudart shared -g --ptxas-options=-v

va_c.o: ../Samples/Copy/va.cu
	nvcc -c ../Samples/Copy/va.cu -o va_c.o --cudart shared -g --ptxas-options=-v

mm_c.o: ../Samples/Copy/mm.cu
	nvcc -c ../Samples/Copy/mm.cu -o mm_c.o --cudart shared -g --ptxas-options=-v

mem_zc.o: ../Samples/ZeroCopy/mem.cu
	nvcc -c ../Samples/ZeroCopy/mem.cu -o mem_zc.o --cudart shared -g --ptxas-options=-v

va_zc.o: ../Samples/ZeroCopy/va.cu
	nvcc -c ../Samples/ZeroCopy/va.cu -o va_zc.o --cudart shared -g --ptxas-options=-v

mm_zc.o: ../Samples/ZeroCopy/mm.cu
	nvcc -c ../Samples/ZeroCopy/mm.cu -o mm_zc.o --cudart shared -g --ptxas-options=-v

clean:
	rm -rf *.o launcher_va_c benchmark_va_c benchmark_va_zc benchmark_mm_c benchmark_mm_zc benchmark_mem_c benchmark_mem_zc bin
