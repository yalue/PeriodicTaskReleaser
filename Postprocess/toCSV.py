#input from stdin 
#output to stdout csv for excel chart

import sys

# skip first line
sys.stdin.readline()
line = sys.stdin.readline()
if line.find('-') == -1:
  print("invalid input")
while True:
  # start of output set
  numbers = sys.stdin.readline().split(' ')
  datasize = int(numbers[1])
  times = []
  print(datasize, end=",")
  while True:
    run = sys.stdin.readline();
    if (run.find('--') > -1):
      break
    if (run == ''):
      exit(0)
    if (run.find('Execution') > -1):
      execution = int(run.split('Execution ')[1].split(' ms')[0])
      times.append(execution)
      print(execution, end=",")
  print()





