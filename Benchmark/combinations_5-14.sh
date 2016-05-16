#!/bin/bash

# Configuration
duration=$((12*60)) # 12 minutes
size=$((2^22))
randsleep="--randsleep"
copy="c"

args="$copy $duration $size $randsleep"

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

echo "4/
4_fasthog/
1_sd_3_fasthog/
2_sd_2_fasthog/
3_sd_1_fasthog/
4_sd/
1_convbench_3_fasthog/
1_convbench_1_sd_2_fasthog/
1_convbench_2_sd_1_fasthog/
1_convbench_3_sd/
2_convbench_2_fasthog/
2_convbench_1_sd_1_fasthog/
2_convbench_2_sd/
3_convbench_1_fasthog/
3_convbench_1_sd/
4_convbench/" | ./run_benchmark_parse.sh $args

echo "MM/
1_mm_1_sd/
2_mm_1_sd/
3_mm_1_sd/
1_mm_1_convbench/
2_mm_1_convbench/
3_mm_1_convbench/
1_mm_1_fasthog/
2_mm_1_fasthog/
3_mm_1_fasthog/" | ./run_benchmark_parse.sh $args

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

echo "VA_CPU_GPU/
1_va_1_sd/
1_va_2_sd/
1_va_3_sd/
1_va_1_convbench/
1_va_2_convbench/
1_va_3_convbench/
1_va_1_fasthog/
1_va_2_fasthog/
1_va_3_fasthog/" | ./run_benchmark_parse.sh $args
