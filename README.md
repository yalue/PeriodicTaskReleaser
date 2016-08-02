PeriodicTaskReleaser
====================

About
-----

This is a collection of CUDA programs intended to measure interference between
GPU processes. It was created as part of ongoing research in real-time systems
at UNC Chapel Hill. Paper: http://cs.unc.edu/~anderson/papers/ospert16.pdf .

Usage
-----

This program requires that CUDA is installed, and the CUDA samples are
available. Until things are cleaned up, you will need to update the include
directories in Samples/Makefile and Benchmark/Makefile to point to the CUDA
samples includes.

The benchmarks can be compiled by navigating to the `Benchmark` directory and
running `make`. Run the benchmarks by running the `combinations_5-14.sh` script
in the `Benchmark` directory. This file can be edited to adjust which
benchmarks are exuected, and a few other parameters.

When benchmarks have finished running, output files will be generated in C/ and
ZC/ directories, containing, respectively, copy and zero-copy results (if both
were run).
