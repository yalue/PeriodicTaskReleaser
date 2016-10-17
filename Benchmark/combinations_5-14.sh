#!/bin/bash

# Configuration

# The amount of time each benchmark runs, in seconds
duration=$((10 * 60))

# The size, in *number of elements*, of the vector or matrices used for the
# vector add or matrix multiply benchmarks.
size=$((2^22))

# Omit this to remove the random sleep between benchmark iterations.
randsleep="--randsleep"

# This can be "c", "zc" or "c zc" to select whether to run copy, zero-copy, or
# both versions of each benchmark.
copy="c"

args="$copy $duration $size $randsleep"

# Benchmarks are invoked using run_benchmark_parse.sh. The first line must
# contain the total number of benchmarks in each scenario, and the following
# lines contain the scenarios themselves.

# These example lines test co-scheduling up to 4 instances of stereo disparity.
echo "1/
1_sd/" | ./run_benchmark_parse.sh $args
echo "2/
2_sd/" | ./run_benchmark_parse.sh $args
echo "3/
3_sd/" | ./run_benchmark_parse.sh $args
echo "4/
4_sd/" | ./run_benchmark_parse.sh $args


# Uncomment this block to run all benchmarks.
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
END
