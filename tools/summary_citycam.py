import os.path as op
from glob import glob
from pprint import pprint
import re
import numpy as np

def parse_experiment_name(x):
  def parse_num(x):
    if x[-1] == 'K':
      return 1000 * int(x[:-1])
    else:
      return int(x)
  parts = x.replace('+','-').split('-')[1:]
  nums = {}
  
  # parse synth and real
  for i,part in enumerate(parts):
    if 'real' in part:
      nums['real']  = parse_num(part.split('real')[0])
    elif 'synth' in part:
      nums['synth'] = parse_num(part.split('synth')[0])
    elif i==len(parts)-1 and not part.isdigit():
      return None

  # blank synth and real
  if 'real' not in nums:
    nums['real'] = 0
  if 'synth' not in nums:
    nums['synth'] = 0

  return nums['synth'], nums['real']


model_dirs = sorted(glob(op.join('output/572-Feb23-09h', '*')))
#pprint(model_dirs)

results = {}

for model_dir in model_dirs:

  result_paths = sorted(glob(op.join(model_dir, 'detected/*.txt')))
  #pprint(result_paths)

  if len(result_paths) == 0:
    print ('model_dir is empty: %s' % model_dir)
    continue
  result_path = result_paths[-1]

  with open(result_path) as f:
    lines = f.read().splitlines()
  for i,line in enumerate(lines):
    if line == 'car_constraint: width > 25':
      name = parse_experiment_name(model_dir)
      if name is None: break
      mAP = float(lines[i+1]) * 100  # in perc.
      if name not in results:
        results[name] = []
      results[name].append(mAP)
      break
  
output = []
#pprint(results)
for name in results:
  row = (name[0], name[1], np.mean(results[name]), np.std(results[name]))
  output.append(row)

output = np.array(output,
    dtype=[('synth', 'i4'),('real', 'i4'), ('mean', 'f2'), ('std', 'f2')])
output.sort(order=('real','synth'))
#np.set_printoptions(precision=2, linewidth=30)
output = np.array_str(output, max_line_width=30, precision=2)
print re.sub('[,()]', '', output)

