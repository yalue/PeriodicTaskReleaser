#!/bin/bash

duration=$((30*60)) #30 minutes
size=$((2^18))

# CPU programs
out="cpu"
mkdir -p $out
echo $out
for sample in va mm
do
  ./run_experiment.sh ./benchmark_${sample}_cpu ${duration} ${size} ${out}/${sample}
done

# Multiple CPU programs
out="2cpu"
mkdir -p $out
echo $out
for sample in va mm
do
  ./run_experiment.sh ./benchmark_${sample}_cpu ${duration} ${size} ${out}/${sample}_1 --randsleep &
  pids[1]=$!
  ./run_experiment.sh ./benchmark_${sample}_cpu ${duration} ${size} ${out}/${sample}_2 --randsleep &
  pids[2]=$!
  for pid in ${pids[@]}
  do
    wait $pid
  done
done

# Four CPU programs
out="4cpu"
mkdir -p $out
echo $out
i=0
for sample in va mm
do
  ./run_experiment.sh ./benchmark_${sample}_cpu ${duration} ${size} ${out}/${sample}_1 --randsleep &
  pids[$i]=$!
  i=$(($i+1))
  ./run_experiment.sh ./benchmark_${sample}_cpu ${duration} ${size} ${out}/${sample}_2 --randsleep &
  pids[$i]=$!
  i=$(($i+1))
done
for pid in ${pids[@]}
do
  wait $pid
done

# GPU programs
out="gpu"
mkdir -p $out 
echo $out
for copy in c zc
do
  for sample in sd sf fasthog convbench 
  do
    ./run_experiment.sh ./benchmark_${sample}_${copy} ${duration} ${size} ${out}/${sample}_${copy}
  done
done

# Four GPU programs
out="4gpu"
mkdir -p $out 
echo $out
for copy in c zc
do
  i=0
  for sample in sd sf fasthog convbench 
  do
    ./run_experiment.sh ./benchmark_${sample}_${copy} ${duration} ${size} ${out}/${sample}_${copy} --randsleep &
    pids[$i]=$!
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
echo $out
for copy in c zc
do
  i=0
  for sample in sd sf fasthog convbench 
  do
    ./run_experiment.sh ./benchmark_${sample}_${copy} ${duration} ${size} ${out}/${sample}_${copy} --randsleep &
    pids[$i]=$!
    i=$(($i+1))
  done
  for sample in va mm
  do
    ./run_experiment.sh ./benchmark_${sample}_cpu ${duration} ${size} ${out}/${sample}_${copy} --randsleep &
    pids[$i]=$!
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
echo $out
for copy in c zc
do
  i=0
  for sample in sd sf fasthog convbench 
  do
    ./run_experiment.sh ./benchmark_${sample}_${copy} ${duration} ${size} ${out}/${sample}_${copy} --randsleep &
    pids[$i]=$!
    i=$(($i+1))
  done
  for sample in va mm
  do
    ./run_experiment.sh ./benchmark_${sample}_cpu ${duration} ${size} ${out}/${sample}_${copy}_1 --randsleep &
    pids[$i]=$!
    i=$(($i+1))
    ./run_experiment.sh ./benchmark_${sample}_cpu ${duration} ${size} ${out}/${sample}_${copy}_2 --randsleep &
    pids[$i]=$!
    i=$(($i+1))
  done
  for pid in ${pids[@]}
  do
    wait $pid
  done
done

