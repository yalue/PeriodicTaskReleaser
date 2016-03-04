import csv
import os
import statistics
import sys

ename = str(sys.argv[1])
inputSize = int(sys.argv[2])
outputName = str(sys.argv[3])

path_to_files = '/home/ubuntu/GPUSync/Experiments/Periodic/Postprocess/' + ename + '/'

listing = os.listdir(path_to_files)

runtimes0 = []
runtimes1 = []
for folder in listing:
  folder_listing = os.listdir(path_to_files + folder)
  for fname in folder_listing:
    file = path_to_files + folder + '/' + fname 
    with open(file) as csvfile:
      reader = csv.reader(csvfile)
      for row in reader:
        if (int(row[0]) == inputSize):
          for time in row[1:]:
            if time != '':
              if '1' in fname:
                runtimes0.append(int(time))
              else:
                runtimes1.append(int(time))
with open(outputName + str(1) + '.csv', 'w+') as out:
  for time in runtimes0:
    out.write("{},\n".format(time));

with open(outputName + str(2) + '.csv', 'w+') as out:
  for time in runtimes1:
    out.write("{},\n".format(time));
