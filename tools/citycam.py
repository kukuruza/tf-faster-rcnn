from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from trainval_net import main as trainval_net
from test_net import main as test_net
from reval import main as reval
from utils.loggers import setup_logging
from model.config import get_experiment

import logging
import argparse
import re
import pprint
import sys, os, os.path as op
from glob import glob
sys.path.insert(0, op.join(os.getenv('CITY_PATH'), 'src'))
from learning.helperSetup import atcity


def parse_args(arg_list):
  parser = argparse.ArgumentParser(description='Execute train/test pipeline for Citycam.')
  parser.add_argument('--gpu', default=0, type=int)
  parser.add_argument('--train_db', required=True,
            help='''path of .db file or its dir, relative to faster-rcnn dir.
                    E.g. 572-Feb23-09h/mar31-synth/my_train_data.db
                    OR   572-Feb23-09h/mar31-synth (assumed mar31-synth.db)''')
  parser.add_argument('--test_db', required=True,
            help='''path of .db file, relative to faster-rcnn dir.
                    E.g. 572-Feb23-09h/labelled-test-e01.db''')
  parser.add_argument('--do', nargs='+', choices=['train', 'test', 'eval'], required=True,
            help='perform training, or testing, or both')

  args, args_list_remaining = parser.parse_known_args(arg_list)
  return args, args_list_remaining


def main(args_list):
  args, args_list_remaining = parse_args(arg_list)

  # set GPU
  os.putenv('CUDA_VISIBLE_DEVICES', str(args.gpu))

  # experiment
  experiment = get_experiment(args.train_db)
  logging.info('experiment: %s' % experiment)

  # model_dir
  model_dir = op.join('output', experiment)

  # test_db_path
  test_db_path = atcity(op.join('data/faster-rcnn', args.test_db))
  assert op.exists(test_db_path), 'test_db_path does not exist: %s' % test_db_path

  if 'train' in args.do:

    # train_db_path
    train_db_path = atcity(op.join('data/faster-rcnn', train_db))
    assert op.exists(train_db_path), train_db_path

    trainval_net (['--train_db_path', train_db_path,
                   '--val_db_path',   test_db_path,
                   '--model_dir',     model_dir] +
                  list(args_list_remaining))

  # find all models
  logging.info ('Looking for pattern %s' % op.join(model_dir, '*.ckpt.index'))
  model_paths = glob(op.join(model_dir, '*.ckpt.index'))
  logging.info ('Found %d models to test' % len(model_paths))

  if 'test' in args.do:

    for model_path in sorted(model_paths):
      model_path = model_path[:-6]
      logging.info ('Will test model %s' % model_path)

      iters = int(re.findall(r'\d+', model_path)[-1])
      out_name = '%s-it%06d' % (op.dirname(args.test_db), iters)
      out_db_path  = op.join(model_dir, 'detected/%s.db' % out_name)
     
      mAP = test_net(['--model_path',    model_path,
                      '--out_db_path',   out_db_path,
                      '--gt_db_path',    test_db_path] +
                      list(args_list_remaining))

  if 'eval' in args.do:

    # find all models
    model_paths = glob(op.join(model_dir, '*.ckpt.index'))
    logging.info ('Found %d models to evaluate' % len(model_paths))
    mAPs = {}

    for model_path in sorted(model_paths):
      model_path = model_path[:-6]
      logging.info ('Will evaluate model %s' % model_path)

      iters = int(re.findall(r'\d+', model_path)[-1])
      out_name = '%s-it%06d' % (op.dirname(args.test_db), iters)
      print ('out_name', out_name)
      out_db_path  = op.join(model_dir, 'detected/%s.db' % out_name)
      results_path = op.join(model_dir, 'detected/%s.txt' % out_name) 
      with open(results_path, 'a') as fid:
        fid.write('test_db_path: %s\n' % test_db_path)

      mAPs[model_path] = {}
      for i,car_constraint in enumerate(['width > 25', 'width > 30', 'width > 35']):
        mAP = reval   (['--out_db_path',   out_db_path,
                        '--results_path',  results_path,
                        '--gt_db_path',    test_db_path] +
                       list(args_list_remaining) +
                       ['TEST.CAR_CONSTRAINT', car_constraint])
        mAPs[model_path][car_constraint] = mAP
    pprint.pprint (mAPs)

if __name__ == '__main__':
  arg_list = sys.argv[1:]
  setup_logging(arg_list)
  main(arg_list)

