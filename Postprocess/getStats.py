import csv
import os
import statistics
import sys

ename = str(sys.argv[1])
enum = 100
fname = 'result.csv'

path_to_files = '/home/ubuntu/GPUSync/Experiments/Periodic/Postprocess/' + ename + '/'

inputSizesSet = False
inputSizes = [];
listing = os.listdir(path_to_files)
results = {}

for folder in listing:
  file = path_to_files + folder + '/' + fname
  with open(file) as csvfile:
    reader = csv.reader(csvfile)
    res_num = int(folder[9:])
    results[res_num] = {}
    for row in reader:
      runtimes = []
      for time in row[1:]:
        if time != '':
          runtimes.append(int(time))
      results[res_num][int(row[0])] = {}
      results[res_num][int(row[0])]['mean'] = statistics.mean(runtimes)
      results[res_num][int(row[0])]['max'] = max(runtimes)
      results[res_num][int(row[0])]['min'] = min(runtimes)
      if inputSizesSet == False:
        inputSizes.append(int(row[0]))
    inputSizesSet = True

print("Datasize, Mean, Max, Min")
for j in inputSizes:
  means = []  
  maxes = [] 
  mins = []
  for i in range(1, enum + 1):
    means.append(results[i][j]['mean'])
    maxes.append(results[i][j]['max'])
    mins.append(results[i][j]['min'])
  print("{}, {}, {}, {}".format(j, statistics.mean(means), max(maxes), min(mins)))
