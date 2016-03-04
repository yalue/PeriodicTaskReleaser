#!/bin/bash

for size in 524288 #1048576 2097152
do
#  python3 bigCSV.py MM/Locks $size > MMLocks${size}.csv
#  python3 bigCSV.py MM/NoLocks $size > MMNoLocks${size}.csv
  python3 bigCSV.py VAMM_AVG/Locks $size VAMM_AVG_LOCKS
#  python3 bigCSV.py VAMM_AVG/NoLocks $size VAMM_AVGNoLocks
done

