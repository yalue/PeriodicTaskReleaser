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
        ./run_experiment.sh ./benchmark_${tok}_cpu $duration $size $out/$line/${tok}_cpu $randsleep &
      else
        ./run_experiment.sh ./benchmark_${tok}_${copy} $duration $size $out/$line/$tok $randsleep &
      fi
      pids[$j]=$!
      j=$(($j+1))

      for i in $(seq 2 $num)
      do
        if [[ "$tok" == "va" ]]
        then
          ./run_experiment_no_log.sh ./benchmark_${tok}_cpu $duration $size $out/$line/${tok}_cpu $randsleep &
        else
          ./run_experiment_no_log.sh ./benchmark_${tok}_${copy} $duration $size $out/$line/$tok $randsleep &
        fi
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
