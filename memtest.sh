operations=("alloc" "copyin" "copyout" "free")
minexp=$((10))
maxexp=$((20))

for operation in `seq 2 3`;
do
  exp=$minexp
  for exp in `seq $minexp $maxexp`;
  do
    echo memtest $((2**$exp)) $operation
    ./memtest $((2**$exp)) $operation > Benchmark/mem/${operations[$((${operation} - 1))]}${exp}.csv
  done
done
