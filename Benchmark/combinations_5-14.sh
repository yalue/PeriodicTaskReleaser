#!/bin/bash

# Configuration
duration=$((10))
size=$((2^22))
randsleep="--randsleep"
copy="c"

args="$copy $duration $size $randsleep"

: <<'END'
# Experiments
echo "1/
1_fasthog/
1_sd/
1_convbench/" | ./run_benchmark_parse.sh $args

echo "2/
2_fasthog/
1_sd_1_fasthog/
2_sd/
1_convbench_1_fasthog/
1_convbench_1_sd/
2_convbench/" | ./run_benchmark_parse.sh $args

echo "3/
3_fasthog/
1_sd_2_fasthog/
2_sd_1_fasthog/
3_sd/
1_convbench_2_fasthog/
1_convbench_1_sd_1_fasthog/
1_convbench_2_sd/
2_convbench_1_fasthog/
2_convbench_1_sd/
3_convbench/" | ./run_benchmark_parse.sh $args
END

echo "4/
2_sd_2_fasthog/" | ./run_benchmark_parse.sh $args

: <<'END'
echo "MM/
3_mm_1_sd/" | ./run_benchmark_parse.sh $args

# Vector add is on CPU by default
echo "VA_CPU/
1_va_1_sd/
2_va_1_sd/
3_va_1_sd/
1_va_1_convbench/
2_va_1_convbench/
3_va_1_convbench/
1_va_1_fasthog/
2_va_1_fasthog/
3_va_1_fasthog/" | ./run_benchmark_parse.sh $args
END
