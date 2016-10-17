SAMPLE_DIR = ../Samples
UTIL_DIR = ../util
INC = -I$(UTIL_DIR) -I$(SAMPLE_DIR)
HOSTARCH = $(shell uname -m)
NVCC_FLAGS := -arch=sm_53 --cudart shared -g --ptxas-options=-v

all: benchmark_va_c benchmark_va_zc benchmark_mm_c benchmark_mm_zc benchmark_sd_c \
		benchmark_sd_zc benchmark_fasthog_c benchmark_fasthog_zc benchmark_va_cpu \
		benchmark_mm_cpu benchmark_convbench_c benchmark_convbench_zc

benchmark_va_c: benchmark.o util.o
	cd $(SAMPLE_DIR) && make va_c.o
	nvcc benchmark.o $(SAMPLE_DIR)/va_c.o util.o -o benchmark_va_c $(NVCC_FLAGS)

benchmark_va_zc: benchmark.o util.o
	cd $(SAMPLE_DIR) && make va_zc.o
	nvcc benchmark.o $(SAMPLE_DIR)/va_zc.o util.o -o benchmark_va_zc $(NVCC_FLAGS)

benchmark_va_cpu: benchmark.o util.o
	cd $(SAMPLE_DIR) && make va_cpu.o
	nvcc benchmark.o $(SAMPLE_DIR)/va_cpu.o util.o -o benchmark_va_cpu $(NVCC_FLAGS)

benchmark_mm_c: benchmark.o util.o
	cd $(SAMPLE_DIR) && make mm_c.o
	nvcc benchmark.o $(SAMPLE_DIR)/mm_c.o util.o -o benchmark_mm_c $(NVCC_FLAGS)

benchmark_mm_zc: benchmark.o util.o
	cd $(SAMPLE_DIR) && make mm_zc.o
	nvcc benchmark.o $(SAMPLE_DIR)/mm_zc.o util.o -o benchmark_mm_zc $(NVCC_FLAGS)

benchmark_mm_cpu: benchmark.o util.o
	cd $(SAMPLE_DIR) && make mm_cpu.o
	nvcc benchmark.o $(SAMPLE_DIR)/mm_cpu.o util.o -o benchmark_mm_cpu $(NVCC_FLAGS) -lm

benchmark_sd_c: benchmark.o util.o
	cd $(SAMPLE_DIR) && make sd_c.o
	nvcc benchmark.o $(SAMPLE_DIR)/sd_c.o util.o -o benchmark_sd_c $(NVCC_FLAGS)

benchmark_sd_zc: benchmark.o util.o
	cd $(SAMPLE_DIR) && make sd_zc.o
	nvcc benchmark.o $(SAMPLE_DIR)/sd_zc.o util.o -o benchmark_sd_zc $(NVCC_FLAGS)

benchmark_fasthog_c: benchmark.o util.o
	cd $(SAMPLE_DIR) && make fasthog_c.a
	nvcc benchmark.o $(SAMPLE_DIR)/fasthog_c.a $(SAMPLE_DIR)/Copy/FastHOG/libs/libfreeimage_$(HOSTARCH).a \
		util.o $(NVCC_FLAGS) -o benchmark_fasthog_c

benchmark_fasthog_zc: benchmark.o util.o
	cd $(SAMPLE_DIR) && make fasthog_zc.a
	nvcc benchmark.o $(SAMPLE_DIR)/fasthog_zc.a $(SAMPLE_DIR)/Copy/FastHOG/libs/libfreeimage_$(HOSTARCH).a \
		util.o $(NVCC_FLAGS) -o benchmark_fasthog_zc

benchmark_convbench_c: benchmark.o util.o
	cd $(SAMPLE_DIR) && make convbench_c.a
	nvcc benchmark.o $(SAMPLE_DIR)/convbench_c.a util.o $(NVCC_FLAGS) -o benchmark_convbench_c

benchmark_convbench_zc: benchmark.o util.o
	cd $(SAMPLE_DIR) && make convbench_zc.a
	nvcc benchmark.o $(SAMPLE_DIR)/convbench_zc.a util.o $(NVCC_FLAGS) -o benchmark_convbench_zc

benchmark.o: ./benchmark.c
	gcc -c ./benchmark.c -Wall -g $(INC)

util.o: $(UTIL_DIR)/util.c
	gcc -c $(UTIL_DIR)/util.c -Wall -g

clean:
	rm -rf *.o benchmark_*[_z]c bin benchmark_*_cpu
	cd $(SAMPLE_DIR) && make clean
