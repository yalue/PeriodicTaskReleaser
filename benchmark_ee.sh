#!/bin/bash

for exp in `seq 10 20`;
do
#  ./benchmark_ee $((2**$exp)) > Benchmark/SCHED_OTHER/MM${exp}.csv
  ./benchmark_ee $((2**$exp)) 1 > Benchmark/SCHED_OTHER/VA${exp}.csv
done

#sudo chrt -f 10 ./benchmark_ee 16384 > Benchmark/SCHED_FIFO/MM14.csv
#sudo chrt -f 10 ./benchmark_ee 262144 1 > Benchmark/SCHED_FIFO/VA18.csv
#sudo chrt -f 10 ./benchmark_ee 32768 > Benchmark/SCHED_FIFO/MM15.csv
#sudo chrt -f 10 ./benchmark_ee 524288 1 > Benchmark/SCHED_FIFO/VA19.csv
#sudo chrt -f 10 ./benchmark_ee 65536 > Benchmark/SCHED_FIFO/MM16.csv
#sudo chrt -f 10 ./benchmark_ee 1048576 1 > Benchmark/SCHED_FIFO/VA20.csv
#sudo chrt -f 10 ./benchmark_ee 131072 > Benchmark/SCHED_FIFO/MM17.csv
#sudo chrt -f 10 ./benchmark_ee 2097152 1 > Benchmark/SCHED_FIFO/VA21.csv
#sudo chrt -f 10 ./benchmark_ee 262144 > Benchmark/SCHED_FIFO/MM18.csv
#sudo chrt -f 10 ./benchmark_ee 4194304 1 > Benchmark/SCHED_FIFO/VA22.csv
#sudo chrt -f 10 ./benchmark_ee 524288 > Benchmark/SCHED_FIFO/MM19.csv
#sudo chrt -f 10 ./benchmark_ee 8388608 1 > Benchmark/SCHED_FIFO/VA23.csv
