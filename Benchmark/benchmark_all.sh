#!/bin/bash

duration=2 #$((30*60)) #30 minutes
sync=0 #spin
#Samples are sd, sf, fasthog
out="benchmark"
mkdir -p $out 
for copy in c zc
do
  for sample in sd sf fasthog mm
  do
    echo $sample $copy $config
    stdbuf -oL ./benchmark_${sample}_${copy} --sync ${sync} --duration ${duration} --size 262144 --${config} > ${out}/${sample}_${copy}.csv &
    pid=$!

    ./cpu_log.sh $pid ${out}/${sample}_${copy}_cpu.txt ./benchmark_${sample}_${copy}
    wait $pid
  done
done

# now four concurrent processes
out="parallel_benchmark/${config}"
mkdir -p $out 
for copy in c zc
do
  i=0
  for sample in sd sf fasthog mm
  do
    echo $sample $copy $config parallel
    stdbuf -oL ./benchmark_${sample}_${copy} --sync ${sync} --duration ${duration} --size 262144 --${config} > ${out}/${sample}_${copy}.csv &
    pids[$i]=$!

    ./cpu_log.sh $! ${out}/${sample}_${copy}_cpu.txt ./benchmark_${sample}_${copy} &
 
    i=$(($i+1))
  done
  for pid in $pids
  do
    wait $pid
  done
done

