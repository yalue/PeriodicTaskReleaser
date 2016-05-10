SAMPLE_DIR = ../Samples
UTIL_DIR = ../util
INC = -I$(UTIL_DIR) -I$(SAMPLE_DIR)
HOSTARCH = armv7l

all: benchmark_va_c benchmark_va_zc benchmark_mm_c benchmark_mm_zc benchmark_mem_c \
		benchmark_mem_zc benchmark_sf_c benchmark_sf_zc benchmark_sd_c benchmark_sd_zc \
		benchmark_fasthog_c benchmark_fasthog_zc benchmark_convbench_c benchmark_convbench_zc

benchmark_va_c: benchmark.o util.o
	cd $(SAMPLE_DIR) && make va_c.o
	nvcc benchmark.o $(SAMPLE_DIR)/va_c.o util.o -o benchmark_va_c --cudart shared -g

benchmark_va_zc: benchmark.o util.o
	cd $(SAMPLE_DIR) && make va_zc.o
	nvcc benchmark.o $(SAMPLE_DIR)/va_zc.o util.o -o benchmark_va_zc --cudart shared -g

benchmark_mm_c: benchmark.o util.o
	cd $(SAMPLE_DIR) && make mm_c.o
	nvcc benchmark.o $(SAMPLE_DIR)/mm_c.o util.o -o benchmark_mm_c --cudart shared -g

benchmark_mm_zc: benchmark.o util.o
	cd $(SAMPLE_DIR) && make mm_zc.o
	nvcc benchmark.o $(SAMPLE_DIR)/mm_zc.o util.o -o benchmark_mm_zc --cudart shared -g

benchmark_mem_c: benchmark.o util.o
	cd $(SAMPLE_DIR) && make mem_c.o
	nvcc benchmark.o $(SAMPLE_DIR)/mem_c.o util.o -o benchmark_mem_c --cudart shared -g

benchmark_mem_zc: benchmark.o util.o
	cd $(SAMPLE_DIR) && make mem_zc.o
	nvcc benchmark.o $(SAMPLE_DIR)/mem_zc.o util.o -o benchmark_mem_zc --cudart shared -g

benchmark_sf_c: benchmark.o util.o
	cd $(SAMPLE_DIR) && make sf_c.o
	nvcc benchmark.o $(SAMPLE_DIR)/sf_c.o util.o -o benchmark_sf_c --cudart shared -g

benchmark_sf_zc: benchmark.o util.o
	cd $(SAMPLE_DIR) && make sf_zc.o
	nvcc benchmark.o $(SAMPLE_DIR)/sf_zc.o util.o -o benchmark_sf_zc --cudart shared -g

benchmark_sd_c: benchmark.o util.o
	cd $(SAMPLE_DIR) && make sd_c.o
	nvcc benchmark.o $(SAMPLE_DIR)/sd_c.o util.o -o benchmark_sd_c --cudart shared -g

benchmark_sd_zc: benchmark.o util.o
	cd $(SAMPLE_DIR) && make sd_zc.o
	nvcc benchmark.o $(SAMPLE_DIR)/sd_zc.o util.o -o benchmark_sd_zc --cudart shared -g

benchmark_fasthog_c: benchmark.o util.o
	cd $(SAMPLE_DIR) && make fasthog_c.a
	nvcc benchmark.o $(SAMPLE_DIR)/fasthog_c.a $(SAMPLE_DIR)/Copy/FastHOG/libs/libfreeimage_$(HOSTARCH).a \
		util.o --cudart shared -g -o benchmark_fasthog_c

benchmark_fasthog_zc: benchmark.o util.o
	cd $(SAMPLE_DIR) && make fasthog_zc.a
	nvcc benchmark.o $(SAMPLE_DIR)/fasthog_zc.a $(SAMPLE_DIR)/Copy/FastHOG/libs/libfreeimage_$(HOSTARCH).a \
		util.o --cudart shared -g -o benchmark_fasthog_zc

benchmark_convbench_c: benchmark.o util.o
	cd $(SAMPLE_DIR) && make convbench_c.a
	nvcc benchmark.o $(SAMPLE_DIR)/convbench_c.a util.o --cudart shared -g -o benchmark_convbench_c

benchmark_convbench_zc: benchmark.o util.o
	cd $(SAMPLE_DIR) && make convbench_zc.a
	nvcc benchmark.o $(SAMPLE_DIR)/convbench_zc.a util.o --cudart shared -g -o benchmark_convbench_zc

benchmark.o: ./benchmark.c
	gcc -c ./benchmark.c -Wall -g $(INC)

memtest.o: ./memtest.c
	gcc -c ./memtest.c -Wall -g

util.o: $(UTIL_DIR)/util.c
	gcc -c $(UTIL_DIR)/util.c -Wall -g

clean:
	rm -rf *.o benchmark_*[_z]c bin
	cd $(SAMPLE_DIR) && make clean
