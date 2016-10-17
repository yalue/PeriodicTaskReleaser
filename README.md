PeriodicTaskReleaser
====================

About
-----

This is a collection of CUDA programs intended to measure interference between
GPU processes. It was created as part of ongoing research in real-time systems
at UNC Chapel Hill. Paper: http://cs.unc.edu/~anderson/papers/ospert16.pdf .

Abbreviations
-------------

Benchmarks are often referred to as shorthand throughout this repository and
associated papers. "SD" refers to Stereo Disparity, "HOG" refers to fastHOG,
"MM" refers to matrix multiply, and "VA" refers to vector add.

Basic Usage
-----------

This program requires that CUDA is installed, and the CUDA samples are
available. Until things are cleaned up, you will need to update the include
directories in Samples/Makefile and Benchmark/Makefile to point to the CUDA
samples includes.

The benchmarks can be compiled by navigating to the `Benchmark` directory and
running `make`. This will produce 2 executables for each benchmark: one "copy"
version, ending in `_c`, and one "zero-copy" version ending in `_zc`. Note that
any zero-copy version isn't necessarily fully zero-copy--many benchmarks made
use of texture memory, and we didn't try to convert these to use zero-copy.
There are also versions of MM and VA ending with a `_cpu` postfix, which
carry out all calculations on the CPU rather than the GPU.

To run a single benchmark, just start the executable. A list of arguments to
each executable can be provided by passing the `--help` argument. While
running, the benchmark will print out lines of CSV times.

The `TX-max_perf.sh` script should be run before any "official" benchmarking to
disable frequency scaling, turn on the fan, and set clock rates to a high
value.

Co-scheduling testing
---------------------

The co-scheduling tests carried out in the associated papers were conducted
using the `combinations_5-14.sh` script in the `Benchmarks` directory. This
file can be edited to adjust which benchmarks are executed, and a few other
parameters.

When benchmarks kicked off by the `combinations_5-14` script have finished
running, output files will be generated in c/ and zc/ directories, containing,
respectively, copy and zero-copy results. If only copy or zero-copy benchmarks
were run, then only one or the other directory will be created.

Measuring block times
---------------------

To measure block times for the stereo disparity benchmark, modify
`Samples/Copy/StereoDisparity/sd.cu`. There is a commented-out block of code in
the `CopyOut` function which can be uncommented to print all recorded block
times to stdout. The reported times must be divided by 1.0e5 in order to get
times in seconds. This multiplication is carried out to reduce the number of
leading zeroes, and therefore log sizes, since recording all blocks generates
a large amount of data.
