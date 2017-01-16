#!/bin/bash

copy=$1 # copy/vs zero copy
duration=$2
size=$3
randsleep=$4
read out
out=./${copy}/${out%%/}
mkdir -p $out

while read line
do
  echo $line
  mkdir -p $out/$line

  IFS='_'
  read -a split <<< "$line"
  IFS=''

  # CPU cores will be assigned in rotation between 0 and 3 (but we don't plan
  # to run more than 4 benchmarks at once). The first task will start on core
  # 1, somewhat arbitrarily.
  cpu_core=1
  num=0
  j=0
  for tok in ${split[@]}
  do
    tok=${tok%%/} # remove the / from the suffix
    re='^[0-9]+$'
    if [[ $tok =~ $re ]] # token is a number
    then
      num=$tok
    else
      if [[ "$tok" == "va" ]]
      then
        ./run_experiment.sh ./benchmark_${tok}_cpu $duration $size $out/$line/${tok}_cpu $randsleep $cpu_core &
      else
        ./run_experiment.sh ./benchmark_${tok}_${copy} $duration $size $out/$line/$tok_1  $randsleep $cpu_core &
      fi
      cpu_core=$(($cpu_core + 1))
      cpu_core=$(($cpu_core % 4))
      pids[$j]=$!
      j=$(($j+1))
      for ((i=2; i <= $num; i++))
      do
        if [[ "$tok" == "va" ]]
        then
          ./run_experiment_no_log.sh ./benchmark_${tok}_cpu $duration $size $out/$line/${tok}_cpu $randsleep $cpu_core &
        else
          ./run_experiment.sh ./benchmark_${tok}_${copy} $duration $size $out/$line/$tok_${i} $randsleep $cpu_core &
        fi
        cpu_core=$(($cpu_core + 1))
        cpu_core=$(($cpu_core % 4))
        pids[$j]=$!
        j=$(($j+1))
      done
    fi
  done
  for pid in ${pids[@]}
  do
    echo "waiting on $pid"
    wait $pid
  done
done
