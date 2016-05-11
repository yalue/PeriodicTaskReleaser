#!/bin/bash

duration=$((30*60)) #30 minutes
size=$((2^18))

# CPU programs
out="cpu"
mkdir -p $out
for sample in va mm
do
  echo $sample
  stdbuf -oL ./benchmark_${sample}_cpu --randsleep --duration ${duration} --size ${size} > ${out}/${sample}_cpu.csv &
  pid=$!
  ./cpu_log.sh $pid ${out}/${sample}_cpu_cpu.csv ./benchmark_${sample}_cpu &
  wait $pid
done

# Multiple CPU programs
out="2cpu"
mkdir -p $out
for sample in va mm
do
  for iteration in one two
  do
    echo $sample $iteration
    stdbuf -oL ./benchmark_${sample}_cpu --randsleep --duration ${duration} --size ${size} > ${out}/${sample}_cpu.csv &
    pids[$i]=$!
    ./cpu_log.sh $! ${out}/${sample}_cpu_cpu.csv ./benchmark_${sample}_cpu &
    i=$(($i+1))
  done
  for pid in ${pids[@]}
  do
    wait $pid
  done
done

# Four CPU programs
out="4cpu"
mkdir -p $out
i=0
for sample in va va mm mm
do
  echo $sample $i
  stdbuf -oL ./benchmark_${sample}_cpu --randsleep --duration ${duration} --size ${size} > ${out}/${sample}_cpu.csv &
  pids[$i]=$!
  ./cpu_log.sh $! ${out}/${sample}_cpu_cpu.csv ./benchmark_${sample}_cpu &
  i=$(($i+1))
done
for pid in ${pids[@]}
do
  wait $pid
done

# GPU programs
out="gpu"
mkdir -p $out 
for copy in c zc
do
  for sample in sd sf fasthog convbench 
  do
    echo $sample $copy 
    stdbuf -oL ./benchmark_${sample}_${copy} --randsleep --duration ${duration} --size ${size} > ${out}/${sample}_${copy}.csv &
    pid=$!
    ./cpu_log.sh $pid ${out}/${sample}_${copy}_cpu.csv ./benchmark_${sample}_${copy} &
    wait $pid
  done
done

# Four GPU programs
out="4gpu"
mkdir -p $out 
for copy in c zc
do
  i=0
  for sample in sd sf fasthog convbench 
  do
    echo $sample $copy parallel
    stdbuf -oL ./benchmark_${sample}_${copy} --randsleep --duration ${duration} --size ${size} > ${out}/${sample}_${copy}.csv &
    pids[$i]=$!
    ./cpu_log.sh $! ${out}/${sample}_${copy}_cpu.csv ./benchmark_${sample}_${copy} &
    i=$(($i+1))
  done
  for pid in ${pids[@]}
  do
    wait $pid
  done
done

# Four GPU + 2 CPU
out="4gpu_2cpu"
mkdir -p $out 
for copy in c zc
do
  i=0
  for sample in sd sf fasthog convbench 
  do
    echo $sample $copy parallel
    stdbuf -oL ./benchmark_${sample}_${copy} --randsleep --duration ${duration} --size ${size} > ${out}/${sample}_${copy}.csv &
    pids[$i]=$!
    ./cpu_log.sh $! ${out}/${sample}_${copy}_cpu.csv ./benchmark_${sample}_${copy} &
    i=$(($i+1))
  done
  for sample in va mm
  do
    echo $sample parallel
    stdbuf -oL ./benchmark_${sample}_cpu --randsleep --duration ${duration} --size ${size} > ${out}/${sample}_cpu.csv &
    pids[$i]=$!
    ./cpu_log.sh $! ${out}/${sample}_cpu_cpu.csv ./benchmark_${sample}_cpu &
    i=$(($i+1))
  done
  for pid in ${pids[@]}
  do
    wait $pid
  done
done

# GPU + 4 CPU
out="4gpu_4cpu"
mkdir -p $out 
for copy in c zc
do
  i=0
  for sample in sd sf fasthog convbench 
  do
    echo $sample $copy parallel
    stdbuf -oL ./benchmark_${sample}_${copy} --randsleep --duration ${duration} --size ${size} > ${out}/${sample}_${copy}.csv &
    pids[$i]=$!
    ./cpu_log.sh $! ${out}/${sample}_${copy}_cpu.csv ./benchmark_${sample}_${copy} &
    i=$(($i+1))
  done
  for sample in va va mm mm
  do
    echo $sample parallel
    stdbuf -oL ./benchmark_${sample}_cpu --randsleep --duration ${duration} --size ${size} > ${out}/${sample}_cpu.csv &
    pids[$i]=$!
    ./cpu_log.sh $! ${out}/${sample}_cpu_cpu.csv ./benchmark_${sample}_cpu &
    i=$(($i+1))
  done
  for pid in $pids
  do
    wait $pid
  done
done


