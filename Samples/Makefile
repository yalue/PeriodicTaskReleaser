INC = -I. -I/usr/local/cuda/samples/common/inc
NVCC_FLAGS := -arch=sm_53 --cudart shared -g --ptxas-options=-v

all: va_c.o mm_c.o sd_c.o fasthog_c.a convbench_c.a va_cpu.o mm_cpu.o

va_cpu.o: CPU/va.c
	gcc -c CPU/va.c -o va_cpu.o

mm_cpu.o: CPU/mm.c
	gcc -c CPU/mm.c -o mm_cpu.o

va_c.o: Copy/va.cu
	nvcc -c Copy/va.cu -o va_c.o $(NVCC_FLAGS)

mm_c.o: Copy/mm.cu
	nvcc -c Copy/mm.cu -o mm_c.o $(NVCC_FLAGS)

sd_c.o: Copy/StereoDisparity/sd.cu
	nvcc -c Copy/StereoDisparity/sd.cu -o sd_c.o $(NVCC_FLAGS) $(INC)

fasthog_c.a:
	cd Copy/FastHOG && make fasthog.a
	cp Copy/FastHOG/fasthog.a fasthog_c.a

convbench_c.a:
	cd Copy/convbench && make convbench.a
	cp Copy/convbench/convbench.a convbench_c.a

clean:
	cd Copy/FastHOG && make clean
	cd Copy/convbench && make clean
	rm -f *.o fasthog_c.a convbench_c.a
