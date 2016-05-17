require 'json'

# Validates and appends the total, memory copy, and kernel times to the given
# array. Returns true if OK.
def check_and_append_times(array, total, copy, kernel)
  if total < 0
    puts "Total time less than 0: " + total.to_s
    return false
  end
  if copy < 0
    puts "Copy time less than 0: " + copy.to_s
    return false
  end
  if kernel < 0
    puts "Kernel time less than 0: " + kernel.to_s
    return false
  end
  array << [total, copy, kernel]
  true
end

# Returns an array of arrays of times (in s): [[total, memory copy, kernel]]
def parse_benchmark_lines(lines)
  # Skip the first lines--column titles
  lines.shift
  # Only record 1 PID's records in the file.
  to_return = []
  seen_pid = ""
  total_time = 0.0
  copy_in_time = 0.0
  copy_out_time = 0.0
  kernel_time = 0.0
  lines.each_with_index do |line, line_number|
    columns = line.gsub(/\s+/, "").split(",")
    timestamp = columns[0].to_f
    if (seen_pid != "") && (seen_pid != columns[2])
      next
    else
      seen_pid = columns[2]
    end
    # Skip the program name and pid, join the rest into 1 tag
    tag = columns[3..999].join(" ")
    if tag =~ /start/
      total_time -= timestamp
    elsif tag =~ /end/
      total_time += timestamp
      if !check_and_append_times(to_return, total_time, copy_in_time +
        copy_out_time, kernel_time)
        puts "Error in line " + line_number.to_s
        exit 1
      end
      total_time = 0.0
      copy_in_time = 0.0
      copy_out_time = 0.0
      kernel_time = 0.0
    elsif tag =~ /cudaLaunch.*?call/
      kernel_time -= timestamp
    elsif tag =~ /cudaLaunch.*?return/
      kernel_time += timestamp
    elsif tag =~ /cudamemcpy.*call.*deviceToHost/
      copy_out_time -= timestamp
    elsif tag =~ /cudamemcpy.*return.*deviceToHost/
      copy_out_time += timestamp
    elsif tag =~ /cudamemcpy.*call.*hostToDevice/
      copy_in_time -= timestamp
    elsif tag =~ /cudamemcpy.*return.*hostToDevice/
      copy_in_time += timestamp
    end
  end
  to_return
end

# Takes an array of arrays. Takes the given field from each of the arrays,
# forms a 1-D array of the selected data, then converts it into CDF form.
# Returns an array of 2 arrays consisting respectively of the X coordinates
# (values) of the CDF plot and the y coordinates (ratio <= x).
def get_field_cdf(data, field)
  return [[], []] if data.size == 0
  data = data.map{|a| a[field]}.sort
  total_data = data.size.to_f
  current_min = data[0]
  count = 0.0
  data_list = [data[0]]
  ratio_list = [0.0]
  data.each do |point|
    count += 1.0
    if point > current_min
      data_list << point
      ratio_list << count / total_data
      current_min = point
    end
  end
  data_list << data[-1]
  ratio_list << 1.0
  [data_list, ratio_list]
end

# Takes existing value, ratio data from get_field_cdf(...) and converts values
# into milliseconds and ratios into percentages. Assumes input values are in
# seconds.
def cdf_to_ms_percent(cdf_data)
  new_values = cdf_data[0].map{|p| p * 1000.0}
  percentages = cdf_data[1].map{|p| p * 100.0}
  [new_values, percentages]
end

# Returns the same structure as get_worst_combos(get_combo_datasets()), but
# includes data from the matrix multiply competing workload data.
def get_mm_datasets(field)
  to_return = {}
  all_files = Dir["**/*.csv"]
  all_files.delete_if{|f| f =~ /_cpu\.csv$/}
  all_files.delete_if{|f| f !~ /MM\//}
  all_files.each do |f|
    benchmark = ""
    scenario = ""
    if f =~ /\/(\d+?)_mm.*?\/(.*?)\.csv$/
      scenario = "vs " + $1.to_s + " matrix mult."
      benchmark = $2
    end
    next if benchmark == ""
    # Ignore the matrix multiply times themselves
    next if benchmark == "mm"
    if !to_return.has_key?(benchmark)
      to_return[benchmark] = {}
    end
    this_benchmark = to_return[benchmark]
    all_benchmark_fields = parse_benchmark_file(f)
    # Field 0 = total time, 1 = memory copy, 2 = kernel
    cdf = cdf_to_ms_percent(get_field_cdf(all_benchmark_fields, field))
    this_benchmark[scenario] = cdf
  end
  to_return
end

# Returns the median value in the given CDF
def get_cdf_median(cdf)
  return 0.0 if cdf.size == 0
  index = 0
  while cdf[1][index] < 50.0
    index += 1
  end
  cdf[0][index - 1]
end

# Returns a list of iteration times
def do_mm(size)
  lines = `./benchmark_mm_c -s #{size.to_s} -d 5 2>/dev/null`.split(/\n+/)
  data = parse_benchmark_lines(lines)
  cdf_to_ms_percent(get_field_cdf(data, 0))
end

runs = {}
sizes = [
  16 * 1024,
  32 * 1024,
  64 * 1024,
  80 * 1024,
  128 * 1024,
  180 * 1024,
  256 * 1024,
  400 * 1024,
  512 * 1024,
  800 * 1024,
  1024 * 1024,
  2048 * 1024,
]
sizes.each do |s|
  size_in_k = (s.to_f / 256).to_i.to_s + "KB"
  cdf = do_mm(s)
  median = get_cdf_median(cdf)
  puts "%s matrix: %f median ms/iteration" % [size_in_k, median]
  runs[size_in_k + " matrices"] = cdf
  GC.start
end
experiments = {}
experiments["matrix sizes"] = {}
experiments["matrix sizes"]["matrix multiply"] = runs
File.open("output.json", 'wb') {|f| f.write(JSON.pretty_generate(experiments))}
