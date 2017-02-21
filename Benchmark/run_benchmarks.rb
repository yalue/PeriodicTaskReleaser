#!/usr/bin/ruby
# This file will run a the set of benchmarks in parallel. It will only run on
# Linux, and probably only on bash.

# The directory of this script.
BASE_DIRECTORY = File.expand_path(File.dirname(__FILE__)) + "/"

# The directory in which to put the results and logs.
OUTPUT_DIRECTORY = BASE_DIRECTORY + "results/"

# The size of inputs, for benchmarks that use it (vector add and matrix mul.)
INPUT_SIZE = 2 ** 22

# The number of seconds for which each scenario is run.
DURATION = 10

# Takes a list of benchmark names and runs them, spread across different
# processors. The actual executable run for each benchmark will have a name
# of the form "./benchmark_<benchmark_name>". This function will create
# subdirectories in the OUTPUT_DIRECTORY.
def run_scenario(benchmarks, log_all = true)
  # Outputs = "<total # running>/<specific combination>/<benchmark>/*.csv"
  log_directory = OUTPUT_DIRECTORY + benchmarks.size.to_s + "/"
  combo = benchmarks.sort.each_with_object(Hash.new(0)) {|v, h| h[v] += 1}
  combo = combo.to_a.sort{|a, b| a[0] <=> b[0]}
  combo = combo.map{|v| v[1].to_s + "_" + v[0]}.join("_")
  log_directory += combo + "/"
  cpu_count = `nproc`.to_i
  cpu_core = 1
  benchmark_count = Hash.new(1)
  pids = []
  benchmarks.each do |benchmark|
    executable = BASE_DIRECTORY + "benchmark_#{benchmark}"
    # Determine the log directory and create it if it doesn't exist.
    log_location = log_directory + benchmark + "/"
    `mkdir -p #{log_location}`
    log_location += benchmark_count[benchmark].to_s + ".csv"
    if !log_all && (benchmark_count[benchmark] > 1)
      log_location = "/dev/null"
    end
    # Spawn new processes for each benchmark.
    # TODO: Run vmstat before and after each benchmark program.
    puts "Running #{executable} on CPU #{cpu_core.to_s}"
    pids << Process.fork do
      # Set the CPU core for this process and its children.
      `taskset -c -p #{cpu_core.to_s} $$`
      # Add the arguments to the command, running using stdbuf to skip OS
      # buffering.
      command = "stdbuf -oL #{executable} --size #{INPUT_SIZE.to_s} " +
        "--duration #{DURATION.to_s} --randsleep"
      # Execute the command, redirecting to the log.
      `#{command} > #{log_location}`
    end
    cpu_core = (cpu_core + 1) % cpu_count
    benchmark_count[benchmark] += 1
  end
  pids.each {|pid| Process.wait(pid)}
end

run_scenario(["va", "mm", "fasthog", "sd"], false)
run_scenario(["va", "mm", "mm"])
