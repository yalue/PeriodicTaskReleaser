#!/bin/bash

#Experiment name
ENAM=VAMM

#Result dir
RAW=/home/ubuntu/GPUSync/Experiments/Periodic/Runs/$ENAM
LOCKS=Locks
NOLOCKS=NoLocks

#Postprocessed dir
PROC=/home/ubuntu/GPUSync/Experiments/Periodic/Postprocess/${ENAM}_AVG

mkdir $PROC
mkdir $PROC/$LOCKS
mkdir $PROC/$NOLOCKS

I=0
for D in $RAW/$NOLOCKS*;
do
  I=$((I+1))
  mkdir $PROC/$NOLOCKS/Processed$I
  X=0
  for F in $D/*;
  do
    cat $F | python3 ./toCSV.py !> $PROC/$NOLOCKS/Processed$I/result${X}.csv
    X=$((X+1))
  done
done
I=0
for D in $RAW/$LOCKS*;
do
  I=$((I+1))
  mkdir $PROC/$LOCKS/Processed$I
  X=0
  for F in $D/*;
  do
    cat $F | python3 ./toCSV.py !> $PROC/$LOCKS/Processed$I/result${X}.csv
    X=$((X+1))
  done
done
