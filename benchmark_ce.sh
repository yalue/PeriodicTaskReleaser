#!/bin/bash

for exp in `seq 10 20`;
do
  ./benchmark_ce $((2**$exp)) > Benchmark/SCHED_OTHER/MM${exp}.csv
  ./benchmark_ce $((2**$exp)) > Benchmark/SCHED_OTHER/VA${exp}.csv
done

#for exp in `seq 10 20`;
#do
#  chrt -f 10 ./benchmark_ce $((2**$exp)) > Benchmark/SCHED_FIFO/MM${exp}.csv
#  chrt -f 10 ./benchmark_ce $((2**$exp)) > Benchmark/SCHED_FIFO/VA${exp}.csv
#done

