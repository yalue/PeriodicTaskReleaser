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
  mkdir -p $out/$line

  IFS='_'
  read -a split <<< "$line"
  IFS=''

  num=0
  for tok in ${split[@]}
  do
    tok=${tok%%/} # remove the / from the suffix
    if [ "$tok" -eq "$tok" ] 2>/dev/null # token is a number
    then
      num=$tok
    else
      echo ./run_experiment.sh ./benchmark_${tok}_${copy} $duration $size $out/$line$tok $randsleep 
      ./run_experiment.sh ./benchmark_${tok}_${copy} $duration $size $out/$line/$tok $randsleep 
      for i in $(seq 2 $num)
      do
        ./run_experiment_no_log.sh ./benchmark_${tok}_${copy} $duration $size $out/$line/$tok $randsleep
      done
    fi
  done
done
