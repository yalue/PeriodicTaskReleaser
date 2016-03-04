#!/bin/bash
N=100

for i in `seq 51 $N`;
do
  echo "Run: ${i}."
#  sudo chrt -f 10 ./launcher
#  sudo LD_PRELOAD=/home/ubuntu/GPUSync/Locks/Kernel_Locks/libcudart_wrapper.so chrt -f 10 ./launcher
  sudo LD_PRELOAD=/home/ubuntu/GPUSync/Locks/Kernel_Locks/libcudart_wrapper.so ./launcher -u45 -r2
  #./launcher -u50 -r2
  echo "Run: ${i} done."
  mkdir "Runs/VAMM/Locks$i"
  cp *.txt "Runs/VAMM/Locks$i"
done
