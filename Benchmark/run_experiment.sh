#!/bin/bash

# Runs a benchmark program with random sleep and writes output to outfile.
# Logs CPU and memory usage and writes to output file.

# Argument 1: program to run.
# Argument 2: duration.
# Argument 3: input size.
# Argument 4: output file name.
# Argument 5: random sleep flag (--randsleep).

program=$1
duration=$2
size=$3
outfile=$4
randsleep=$5

vmstat -s > ${outfile}_vmstat_pre.txt
stdbuf -oL ${program} $randsleep --duration ${duration} --size ${size} > ${outfile}.csv &
pid=$!
./cpu_log.sh ${pid} ${outfile}_cpu.csv $program &
wait ${pid}
vmstat -s > ${outfile}_vmstat_post.txt

