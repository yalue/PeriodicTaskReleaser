#input from stdin 
#output to stdout 

import sys
# Header line
print('Iteration, Used, Unused');

while True:
  line = sys.stdin.readline()
  if line == '':
    break
  if line.find('SUMMARY') == -1:
    continue

  split = line.split(',');
 
  print('{}, {}, {}'.format(split[1], split[3], split[5]));

