#!/bin/bash

operations=("alloc" "copyin" "copyout" "free")
minexp=$((10))
maxexp=$((20))

for operation in `seq 1 4`;
do
  exp=$minexp
  for exp in `seq $minexp $maxexp`;
  do
    ./memtest_zc $((2**$exp)) $operation > Benchmark/mem/${operations[$((${operation} - 1))]}${exp}.csv
  done
done
