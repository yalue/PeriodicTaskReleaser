INC = -I/home/ubuntu/NVIDIA_CUDA-6.5_Samples/common/inc -I.
INC += -I/home/ubuntu/samples/NVIDIA_CUDA-6.5_Samples/common/inc

all: mem_c.o va_c.o mm_c.o mem_zc.o va_zc.o mm_zc.o sf_c.o sf_zc.o sd_c.o sd_zc.o fastHOG_c.a

mem_c.o: Copy/mem.cu
	nvcc -c Copy/mem.cu -o mem_c.o --cudart shared -g --ptxas-options=-v

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
	mv Copy/FastHOG/fasthog.a fasthog_c.a

clean:
	cd Copy/FastHOG && make clean
	rm -rf *.o fasthog_c.a bin
