#!/bin/bash

duration=30*60 #30 minutes
sync=0 #spin
#Samples are sd, sf, fasthog
for sample in sd_c sf_c fasthog_c sd_zc sf_zc
do
  all_out="SCHED_OTHER/${sample}/"
  mkdir -p $all_out 
  stdbuf -oL ./benchmark_${sample} -y${sync} -d${duration} --all > $all_out/results.csv
done

