SAMPLE_DIR = ../Samples
UTIL_DIR = ../util
INC = -I$(UTIL_DIR) -I$(SAMPLE_DIR)

all: launcher_va_c launcher_va_zc launcher_mm_c launcher_mm_zc launcher_sf_c launcher_sf_zc launcher_sd_c \
	launcher_sd_zc

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

launcher_sf_c: launcher.o  runner.o util.o
	cd $(SAMPLE_DIR) && make sf_c.o
	nvcc launcher.o $(SAMPLE_DIR)/sf_c.o runner.o util.o --cudart shared -g -o launcher_sf_c -lpthread

launcher_sf_zc: launcher.o runner.o util.o
	cd $(SAMPLE_DIR) && make sf_zc.o
	nvcc launcher.o $(SAMPLE_DIR)/sf_zc.o runner.o util.o --cudart shared -g -o launcher_sf_zc -lpthread

launcher_sd_c: launcher.o  runner.o util.o
	cd $(SAMPLE_DIR) && make sd_c.o
	nvcc launcher.o $(SAMPLE_DIR)/sd_c.o runner.o util.o --cudart shared -g -o launcher_sd_c -lpthread

launcher_sd_zc: launcher.o  runner.o util.o
	cd $(SAMPLE_DIR) && make sd_zc.o
	nvcc launcher.o $(SAMPLE_DIR)/sd_zc.o runner.o util.o --cudart shared -g -o launcher_sd_zc -lpthread

runner.o: runner.c
	gcc -c runner.c -Wall -g $(INC)

launcher.o: launcher.c
	gcc -c launcher.c -Wall -g $(INC)

util.o: $(UTIL_DIR)/util.c
	gcc -c $(UTIL_DIR)/util.c -Wall -g $(INC)

clean:
	rm -rf *.o launcher_va_c launcher_va_zc launcher_mm_c launcher_mm_zc bin launcher_sf_c launcher_sd_c
	cd $(SAMPLE_DIR) && make clean
