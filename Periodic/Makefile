SAMPLE_DIR = ../Samples
UTIL_DIR = ../util
INC = -I$(UTIL_DIR) -I$(SAMPLE_DIR)

all: launcher_va_c launcher_va_zc launcher_mm_c launcher_mm_zc

launcher_mm_zc: launcher.o  runner.o util.o 
	cd $(SAMPLE_DIR) && make mm_zc.o
	nvcc launcher.o $(SAMPLE_DIR)/mm_zc.o runner.o util.o --cudart shared -g -o launcher_mm_zc -lpthread

launcher_mm_c: launcher.o  runner.o util.o
	cd $(SAMPLE_DIR) && make mm_c.o
	nvcc launcher.o $(SAMPLE_DIR)/mm_c.o runner.o util.o --cudart shared -g -o launcher_mm_c -lpthread

launcher_va_zc: launcher.o  runner.o util.o
	cd $(SAMPLE_DIR) && make va_zc.o
	nvcc launcher.o $(SAMPLE_DIR)/va_zc.o runner.o util.o --cudart shared -g -o launcher_va_zc -lpthread

launcher_va_c: launcher.o  runner.o util.o
	cd $(SAMPLE_DIR) && make va_c.o
	nvcc launcher.o $(SAMPLE_DIR)/va_c.o runner.o util.o --cudart shared -g -o launcher_va_c -lpthread

runner.o: runner.c
	gcc -c runner.c -Wall -g $(INC)

launcher.o: launcher.c
	gcc -c launcher.c -Wall -g $(INC)

util.o: $(UTIL_DIR)/util.c
	gcc -c $(UTIL_DIR)/util.c -Wall -g $(INC)

clean:
	rm -rf *.o launcher_va_c launcher_va_zc launcher_mm_c launcher_mm_zc bin
