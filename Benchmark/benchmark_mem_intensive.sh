#!/bin/bash

duration=$((4*60)) #4 minutes
size=$((2^20))

# Fixed GPU + 0-3 VA 2^20
for copy in c zc
do
  for j in `seq 0 3`
  do
    out="gpu_va/${copy}/${sample}/${j}"
    mkdir -p $out 
    echo $out
    i=0
    ./run_experiment.sh ./benchmark_sd_${copy} ${duration} ${size} ${out}/${sample}_${copy} &
    pids[$i]=$!
    i=$(($i+1))
    ./run_experiment.sh ./benchmark_fasthog_${copy} ${duration} ${size} ${out}/${sample}_${copy} &
    pids[$i]=$!
    i=$(($i+1))
    ./run_experiment.sh ./benchmark_convbench_${copy} ${duration} ${size} ${out}/${sample}_${copy} &
    pids[$i]=$!
    i=$(($i+1))
    for k in `seq 0 $j`
    do
      ./run_experiment.sh ./benchmark_va_cpu ${duration} ${size} ${out}/va_cpu_${k} &
      pids[$i]=$!
      i=$(($i+1))
    done
    for pid in ${pids[@]}
    do
      wait $pid
    done
  done
done

