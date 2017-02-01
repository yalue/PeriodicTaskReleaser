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
available. This project assumes that `/usr/local/cuda` is a symlink to the CUDA
installation, including the samples.

The benchmarks can be compiled by navigating to the `Benchmark` directory and
running `make`. This will produce one executable for each benchmark. To execute
a single benchmark, just run its executable. Details about how to use these
executables can be obtained by passing the `--help` argument.

The `TX-max_perf.sh` script will increase the clock rates and turn on the fan
on Jetson TX1 boards. This is useful when you don't want dynamic frequency
adjustments to interfere with benchmarking.

Co-scheduling testing
---------------------

The behavior of multiple benchmarks running together can be tested by running
the `Benchmark/run_benchmarks.rb` ruby script. To run a collection of
benchmarks simultaneously, just add calls to the `run_scenario(..)` function at
the bottom of the script.

When benchmarks have finished running, output files will be generated in the
`Benchmark/results/` directory. This will contain subdirectories indicating the
number of simultaneous benchmarks, the specific collection of benchmarks, and
then each individual benchmark. For example, if `run_benchmarks.rb` contains
the line `run_scenario(["mm", "mm", "va"])`, then the following directory
structure will exist:

```
results/
    3/                 : Results when co-scheduling 3 processes
        2_mm_1_va/     : Results when co-scheduling 2 MM and 1 VA
            va/        : VA benchmark results
                1.csv  : The output of the VA benchmark
            mm/        : MM benchmark results
                1.csv  : The output of the 1st MM benchmark
                2.csv  : The output of the 2nd MM benchmark
```

Measuring block times
---------------------

To measure block times for the stereo disparity benchmark, modify
`Samples/Copy/StereoDisparity/sd.cu`. There is a commented-out block of code in
the `CopyOut` function which can be uncommented to print all recorded block
times to stdout. The reported times must be divided by 1.0e5 in order to get
times in seconds. This multiplication is carried out to reduce the number of
leading zeroes, and therefore log sizes. Recording all blocks generates a large
amount of data.
