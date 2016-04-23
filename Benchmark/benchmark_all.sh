#!/bin/bash
# vector add and mm

iterations=10000

for sample in mem mm va
do
  for memory in c zc
  do  
    for exp in `seq 10 20`;
    do
      size=$((2**$exp))

      all_out="SCHED_OTHER/${sample}/${memory}/all"
      copy_out="SCHED_OTHER/${sample}/${memory}/copy"
      exec_out="SCHED_OTHER/${sample}/${memory}/exec"

      mkdir -p $all_out 
      mkdir -p $copy_out
      mkdir -p $exec_out
      ./benchmark_${sample}_${memory} -s$size -n${iterations} --all > $all_out/${exp}.csv
      ./benchmark_${sample}_${memory} -s$size -n${iterations} --copy > $copy_out/${exp}.csv
      ./benchmark_${sample}_${memory} -s$size -n${iterations} --exec > $exec_out/${exp}.csv
    done
  done
done

