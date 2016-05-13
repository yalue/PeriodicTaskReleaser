#!/bin/bash

duration=$((4*60)) #4 minutes
size=$((2^20))

# CPU programs
for copy in c zc
do
  for sample in sd fasthog convbench 
  do
    out="1gpu/${copy}/${sample}"
    mkdir -p $out 
    echo $out
    ./run_experiment.sh ./benchmark_${sample}_${copy} ${duration} ${size} ${out}/${sample}_${copy} --randsleep
  done
done

# 2 GPU programs
for copy in c zc
do
  for fixed in sd fasthog convbench
  do
    for sample in sd fasthog convbench 
    do
      if [ "$fixed" = "$sample" ]
      then
        continue
      fi
      out="2gpu/${copy}/${fixed}/${sample1}"
      mkdir -p $out 
      echo $out
      i=0
      ./run_experiment.sh ./benchmark_${fixed}_${copy} ${duration} ${size} ${out}/${fixed}_${copy} --randsleep &
      pids[$i]=$!
      i=$(($i+1))
      ./run_experiment.sh ./benchmark_${sample}_${copy} ${duration} ${size} ${out}/${sample}_${copy} --randsleep &
      pids[$i]=$!
      i=$(($i+1))
      for pid in ${pids[@]}
      do
        wait $pid
      done
    done
  done
done

# 3 GPU programs
for copy in c zc
do
  for fixed in sd fasthog convbench
  do
    for sample1 in sd fasthog convbench 
    do
      if [ "$fixed" = "$sample1" ]
      then
        continue
      fi
      for sample2 in sd fasthog convbench
      do
        if [[ ("$fixed" = "$sample2") || ("$sample1" = "$sample2") ]]
        then
          continue
        fi
        out="3gpu/${copy}/${fixed}/${sample1}/${sample2}"
        mkdir -p $out 
        echo $out
        i=0
        ./run_experiment.sh ./benchmark_${fixed}_${copy} ${duration} ${size} ${out}/${fixed}_${copy} --randsleep &
        pids[$i]=$!
        i=$(($i+1))
        ./run_experiment.sh ./benchmark_${sample1}_${copy} ${duration} ${size} ${out}/${sample1}_${copy} --randsleep &
        pids[$i]=$!
        i=$(($i+1))
        ./run_experiment.sh ./benchmark_${sample2}_${copy} ${duration} ${size} ${out}/${sample2}_${copy} --randsleep &
        pids[$i]=$!
        i=$(($i+1))
        for pid in ${pids[@]}
        do
          wait $pid
        done
      done
    done
  done
done

# 4 GPU programs
for copy in c zc
do
  for fixed in sd fasthog convbench
  do
    for sample1 in sd fasthog convbench 
    do
      if [ "$fixed" = "$sample1" ]
      then
        continue
      fi
      for sample2 in sd fasthog convbench
      do
        if [[ ("$fixed" = "$sample2") || ("$sample1" = "$sample2") ]]
        then
          continue
        fi
        for sample3 in sd fasthog convbench
        do
          if [[ ("$fixed" = "$sample3") || ("$sample1" = "$sample3") || ("$sample2" = "$sample3") ]]
          then
            continue
          fi
          out="4gpu/${copy}/${fixed}/${sample1}/${sample2}/${sample3}"
          mkdir -p $out 
          echo $out
          i=0
          ./run_experiment.sh ./benchmark_${fixed}_${copy} ${duration} ${size} ${out}/${fixed}_${copy} --randsleep &
          pids[$i]=$!
          i=$(($i+1))
          ./run_experiment.sh ./benchmark_${sample1}_${copy} ${duration} ${size} ${out}/${sample1}_${copy} --randsleep &
          pids[$i]=$!
          i=$(($i+1))
          ./run_experiment.sh ./benchmark_${sample2}_${copy} ${duration} ${size} ${out}/${sample2}_${copy} --randsleep &
          pids[$i]=$!
          i=$(($i+1))
          ./run_experiment.sh ./benchmark_${sample3}_${copy} ${duration} ${size} ${out}/${sample3}_${copy} --randsleep &
          pids[$i]=$!
          i=$(($i+1))
          for pid in ${pids[@]}
          do
            wait $pid
          done
        done
      done
    done
  done
done



# Fixed GPU + 0-3 MM 2^20
for copy in c zc
do
  for sample in sd fasthog convbench 
  do
    for j in `seq 0 3`
    do
      out="gpu_mm/${copy}/${sample}/${j}"
      mkdir -p $out 
      echo $out
      i=0
      ./run_experiment.sh ./benchmark_${sample}_${copy} ${duration} ${size} ${out}/${sample}_${copy} --randsleep &
      pids[$i]=$!
      i=$(($i+1))
      for k in `seq 1 $j`
      do
        ./run_experiment.sh ./benchmark_mm_${copy} ${duration} ${size} ${out}/mm_${copy}_${k} --randsleep &
        pids[$i]=$!
        i=$(($i+1))
      done
      for pid in ${pids[@]}
      do
        wait $pid
      done
    done
  done
done

