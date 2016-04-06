#!/bin/bash
# vector add and mm
for exp in `seq 10 20`;
do
  ./benchmark_all $((2**$exp)) > Benchmark/SCHED_OTHER/MM${exp}.csv
  ./benchmark_all $((2**$exp)) 1 > Benchmark/SCHED_OTHER/VA${exp}.csv
done

#for exp in `seq 10 20`;
#do
#  sudo chrt -f 10 ./benchmark_all $((2**$exp)) > Benchmark/SCHED_FIFO/MM${exp}.csv
#  sudo chrt -f 10 ./benchmark_all $((2**$exp)) 1 > Benchmark/SCHED_FIFO/VA${exp}.csv
#done

