.PHONY: all clean

NVCC := nvcc -ccbin g++
NVCCFLAGS := --cudart shared -g --ptxas-options=-v
INCLUDES := -I/home/ubuntu/NVIDIA_CUDA-6.5_Samples/common/inc -I.
INCLUDES += -I/home/ubuntu/samples/NVIDIA_CUDA-6.5_Samples/common/inc
INCLUDES += -I/home/ubuntu/NVIDIA_CUDA_7.0/common/inc

all: convbench.a

convbench.o: convbench.cu
	$(NVCC) $(INCLUDES) $(NVCCFLAGS) -o $@ -c $<

matrixMul.o: matrixMul.cu matrixMul.h
	$(NVCC) $(INCLUDES) $(NVCCFLAGS) -o $@ -c $<

im2col.o: im2col.cu im2col.h
	$(NVCC) $(INCLUDES) $(NVCCFLAGS) -o $@ -c $<

convbench.a: im2col.o matrixMul.o convbench.o
	ar rcsv convbench.a *.o
	ranlib convbench.a

clean:
	rm -f *.o convbench.a
