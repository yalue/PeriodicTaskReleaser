SAMPLE_DIR = ../Samples
UTIL_DIR = ../util
INC = -I$(UTIL_DIR) -I$(SAMPLE_DIR)

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
	gcc -c runner.c -Wall -g $(INC)

launcher.o: launcher.c
	gcc -c launcher.c -Wall -g $(INC)

util.o: $(UTIL_DIR)/util.c
	gcc -c $(UTIL_DIR)/util.c -Wall -g $(INC)

mem_c.o: $(SAMPLE_DIR)/Copy/mem.cu
	nvcc -c $(SAMPLE_DIR)/Copy/mem.cu -o mem_c.o --cudart shared -g --ptxas-options=-v

va_c.o: $(SAMPLE_DIR)/Copy/va.cu
	nvcc -c $(SAMPLE_DIR)/Copy/va.cu -o va_c.o --cudart shared -g --ptxas-options=-v

mm_c.o: $(SAMPLE_DIR)/Copy/mm.cu
	nvcc -c $(SAMPLE_DIR)/Copy/mm.cu -o mm_c.o --cudart shared -g --ptxas-options=-v

mem_zc.o: $(SAMPLE_DIR)/ZeroCopy/mem.cu
	nvcc -c $(SAMPLE_DIR)/ZeroCopy/mem.cu -o mem_zc.o --cudart shared -g --ptxas-options=-v

va_zc.o: $(SAMPLE_DIR)/ZeroCopy/va.cu
	nvcc -c $(SAMPLE_DIR)/ZeroCopy/va.cu -o va_zc.o --cudart shared -g --ptxas-options=-v

mm_zc.o: $(SAMPLE_DIR)/ZeroCopy/mm.cu
	nvcc -c $(SAMPLE_DIR)/ZeroCopy/mm.cu -o mm_zc.o --cudart shared -g --ptxas-options=-v

clean:
	rm -rf *.o launcher_va_c launcher_va_zc launcher_mm_c launcher_mm_zc bin
