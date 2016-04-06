all: launcher benchmark_ee benchmark_ce memtest benchmark_all benchmark_all_zc memtest_zc benchmark_ee_zc benchmark_ce_zc

benchmark_all: benchmark_all.o va_c.o mm_c.o util.o
	nvcc benchmark_all.o mm_c.o va_c.o util.o -o benchmark_all --cudart shared -g

benchmark_all_zc: benchmark_all.o va_zc.o mm_zc.o util.o
	nvcc benchmark_all.o mm_zc.o va_zc.o util.o -o benchmark_all_zc --cudart shared -g

benchmark_ee: benchmark_ee.o va_c.o mm_c.o util.o
	nvcc benchmark_ee.o mm_c.o va_c.o util.o -o benchmark_ee --cudart shared -g

benchmark_ee_zc: benchmark_ee.o va_zc.o mm_zc.o util.o
	nvcc benchmark_ee.o mm_zc.o va_zc.o util.o -o benchmark_ee_zc --cudart shared -g

benchmark_ce: benchmark_ce.o va_c.o mm_c.o util.o
	nvcc benchmark_ce.o mm_c.o va_c.o util.o -o benchmark_ce --cudart shared -g

benchmark_ce_zc: benchmark_ce.o va_zc.o mm_zc.o util.o
	nvcc benchmark_ce.o mm_zc.o va_zc.o util.o -o benchmark_ce_zc --cudart shared -g

memtest: memtest.o mem_c.o util.o
	nvcc memtest.o mem_c.o util.o -o memtest --cudart shared -g

memtest_zc: memtest.o mem_zc.o util.o
	nvcc memtest.o mem_zc.o util.o -o memtest_zc --cudart shared -g

benchmark_all.o: Benchmark/benchmark_all.c
	gcc -c Benchmark/benchmark_all.c -Wall -g

benchmark_ee.o: Benchmark/benchmark_ee.c
	gcc -c Benchmark/benchmark_ee.c -Wall -g

benchmark_ce.o: Benchmark/benchmark_ce.c
	gcc -c Benchmark/benchmark_ce.c -Wall -g

memtest.o: Benchmark/memtest.c 
	gcc -c Benchmark/memtest.c -Wall -g

launcher: launcher.o  runner.o util.o mm_c.o va_c.o
	nvcc launcher.o mm_c.o va_c.o runner.o util.o -o launcher -lpthread --cudart shared -g

runner.o: runner.c
	gcc -c runner.c -Wall -g

launcher.o: launcher.c
	gcc -c launcher.c -Wall -g

util.o: util.c
	gcc -c util.c -Wall -g

mem_c.o: Samples/Copy/mem.cu
	nvcc -c Samples/Copy/mem.cu -o mem_c.o --cudart shared -g --ptxas-options=-v

va_c.o: Samples/Copy/va.cu
	nvcc -c Samples/Copy/va.cu -o va_c.o --cudart shared -g --ptxas-options=-v

mm_c.o: Samples/Copy/mm.cu
	nvcc -c Samples/Copy/mm.cu -o mm_c.o --cudart shared -g --ptxas-options=-v

mem_zc.o: Samples/ZeroCopy/mem.cu
	nvcc -c Samples/ZeroCopy/mem.cu -o mem_zc.o --cudart shared -g --ptxas-options=-v

va_zc.o: Samples/ZeroCopy/va.cu
	nvcc -c Samples/ZeroCopy/va.cu -o va_zc.o --cudart shared -g --ptxas-options=-v

mm_zc.o: Samples/ZeroCopy/mm.cu
	nvcc -c Samples/ZeroCopy/mm.cu -o mm_zc.o --cudart shared -g --ptxas-options=-v

clean:
	rm -f *.o launcher benchmark_ce benchmark_ee memtest benchmark_all benchmark_all_zc memtest_zc benchmark_ee_zc benchmark_ce_zc
