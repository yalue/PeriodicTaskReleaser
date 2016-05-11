INC = -I/home/ubuntu/NVIDIA_CUDA-6.5_Samples/common/inc -I.
INC += -I/home/ubuntu/samples/NVIDIA_CUDA-6.5_Samples/common/inc
INC += -I/home/ubuntu/NVIDIA_CUDA_7.0/common/inc

all: mem_c.o va_c.o mm_c.o mem_zc.o va_zc.o mm_zc.o sf_c.o sf_zc.o sd_c.o sd_zc.o \
	fasthog_c.a fasthog_zc.a convbench_c.a convbench_zc.a va_cpu.o mm_cpu.o

mem_c.o: Copy/mem.cu
	nvcc -c Copy/mem.cu -o mem_c.o --cudart shared -g --ptxas-options=-v

va_cpu.o: CPU/va.c
	gcc -c CPU/va.c -o va_cpu.o

mm_cpu.o: CPU/mm.c
	gcc -c CPU/mm.c -o mm_cpu.o

va_c.o: Copy/va.cu
	nvcc -c Copy/va.cu -o va_c.o --cudart shared -g --ptxas-options=-v

mm_c.o: Copy/mm.cu
	nvcc -c Copy/mm.cu -o mm_c.o --cudart shared -g --ptxas-options=-v

sf_c.o: Copy/SobelFilter/sf.cu
	nvcc -c Copy/SobelFilter/sf.cu -o sf_c.o --cudart shared -g --ptxas-options=-v $(INC)

sf_zc.o: ZeroCopy/SobelFilter/sf.cu
	nvcc -c ZeroCopy/SobelFilter/sf.cu -o sf_zc.o --cudart shared -g --ptxas-options=-v $(INC)

sd_c.o: Copy/StereoDisparity/sd.cu
	nvcc -c Copy/StereoDisparity/sd.cu -o sd_c.o --cudart shared -g --ptxas-options=-v $(INC)

sd_zc.o: ZeroCopy/StereoDisparity/sd.cu
	nvcc -c ZeroCopy/StereoDisparity/sd.cu -o sd_zc.o --cudart shared -g --ptxas-options=-v $(INC)

mem_zc.o: ZeroCopy/mem.cu
	nvcc -c ZeroCopy/mem.cu -o mem_zc.o --cudart shared -g --ptxas-options=-v

va_zc.o: ZeroCopy/va.cu
	nvcc -c ZeroCopy/va.cu -o va_zc.o --cudart shared -g --ptxas-options=-v

mm_zc.o: ZeroCopy/mm.cu
	nvcc -c ZeroCopy/mm.cu -o mm_zc.o --cudart shared -g --ptxas-options=-v

fasthog_c.a:
	cd Copy/FastHOG && make fasthog.a
	cp Copy/FastHOG/fasthog.a fasthog_c.a

fasthog_zc.a:
	cd ZeroCopy/FastHOG && make fasthog.a
	cp ZeroCopy/FastHOG/fasthog.a fasthog_zc.a

convbench_c.a:
	cd Copy/convbench && make convbench.a
	cp Copy/convbench/convbench.a convbench_c.a

convbench_zc.a:
	cd ZeroCopy/convbench && make convbench.a
	cp ZeroCopy/convbench/convbench.a convbench_zc.a

clean:
	cd Copy/FastHOG && make clean
	cd ZeroCopy/FastHOG && make clean
	cd Copy/convbench && make clean
	cd ZeroCopy/convbench && make clean
	rm -f *.o fasthog_c.a fasthog_zc.a convbench_c.a convbench_zc.a
