SAMPLE_DIR = ../Samples
INC = -I$(SAMPLE_DIR)
HOSTARCH = $(shell uname -m)
NVCC_FLAGS := -arch=sm_53 --cudart shared -g --ptxas-options=-v

all: benchmark_va_c benchmark_mm_c benchmark_sd_c benchmark_fasthog_c \
	benchmark_va_cpu benchmark_mm_cpu benchmark_convbench_c

benchmark_va_c: benchmark.o
	cd $(SAMPLE_DIR) && make va_c.o
	nvcc benchmark.o $(SAMPLE_DIR)/va_c.o -o benchmark_va_c $(NVCC_FLAGS)

benchmark_va_cpu: benchmark.o
	cd $(SAMPLE_DIR) && make va_cpu.o
	nvcc benchmark.o $(SAMPLE_DIR)/va_cpu.o -o benchmark_va_cpu $(NVCC_FLAGS)

benchmark_mm_c: benchmark.o
	cd $(SAMPLE_DIR) && make mm_c.o
	nvcc benchmark.o $(SAMPLE_DIR)/mm_c.o -o benchmark_mm_c $(NVCC_FLAGS)

benchmark_mm_cpu: benchmark.o
	cd $(SAMPLE_DIR) && make mm_cpu.o
	nvcc benchmark.o $(SAMPLE_DIR)/mm_cpu.o -o benchmark_mm_cpu $(NVCC_FLAGS) -lm

benchmark_sd_c: benchmark.o
	cd $(SAMPLE_DIR) && make sd_c.o
	nvcc benchmark.o $(SAMPLE_DIR)/sd_c.o -o benchmark_sd_c $(NVCC_FLAGS)

benchmark_fasthog_c: benchmark.o
	cd $(SAMPLE_DIR) && make fasthog_c.a
	nvcc benchmark.o $(SAMPLE_DIR)/fasthog_c.a $(SAMPLE_DIR)/Copy/FastHOG/libs/libfreeimage_$(HOSTARCH).a \
		$(NVCC_FLAGS) -o benchmark_fasthog_c

benchmark_convbench_c: benchmark.o
	cd $(SAMPLE_DIR) && make convbench_c.a
	nvcc benchmark.o $(SAMPLE_DIR)/convbench_c.a $(NVCC_FLAGS) -o benchmark_convbench_c

benchmark.o: ./benchmark.c
	gcc -c ./benchmark.c -Wall -g $(INC)

clean:
	rm -f *.o benchmark_*
	cd $(SAMPLE_DIR) && make clean
