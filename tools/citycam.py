from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from trainval_net import main as trainval_net
from test_net import main as test_net
from reval import main as reval
from utils.loggers import setup_logging, setup_config
from model.config import cfg

import logging
import argparse
import re
import pprint
import sys, os, os.path as op
from glob import glob
sys.path.insert(0, op.join(os.getenv('CITY_PATH'), 'src'))
from learning.helperSetup import atcity


def _get_citycam_path(path):
  # can be absolute OR rel. to citycam/data/faster-rcnn OR rel. to citycam
  if op.exists(path):
    return path
  elif op.exists(atcity(path)):
    return atcity(path)
  elif op.exists(atcity(op.join('data/faster-rcnn', path))):
    return atcity(op.join('data/faster-rcnn', path))
  assert 0, '%s not relative to or absolute' % path

def _get_db_file_path(path):
  # can be a dir or a .db file
  if op.isfile(path) and path[-3:] == '.db':
    return path
  elif op.exists(op.join(path, '%s.db' % op.basename(path))):
    return op.join(path, '%s.db' % op.basename(path))
  assert 0, 'cannot infer db_path from %s' % path

def get_train_db_path(train_db):
  train_db = _get_citycam_path (train_db)
  train_db = _get_db_file_path (train_db)
  logging.info('train_db_path: %s' % train_db)
  return train_db

def get_test_db_path(test_db):
  test_db = _get_citycam_path (test_db)
  test_db = _get_db_file_path (test_db)
  logging.info('test_db_path: %s' % test_db)
  return test_db

def get_detected_name(test_db, model_path):
  test_db = _get_citycam_path(test_db)
 
  # get test dir (input can be a dir or a .db file)
  if op.isfile(test_db) and test_db[-3:] == '.db':
    test_db_dir = op.dirname(test_db)
  elif op.isdir(test_db):
    test_db_dir = test_db
  else:
    assert 0, 'cannot infer test_db_dir path from %s' % train_db

  iters = int(re.findall(r'\d+', model_path)[-1])
  detected_name = '%s-it%06d' % (op.basename(test_db_dir), iters)
  logging.info('detected_name: %s' % detected_name)
  return detected_name


def parse_args(arg_list):
  parser = argparse.ArgumentParser(description='Execute train/test pipeline for Citycam.')
  parser.add_argument('--gpu', default=0, type=int)
  parser.add_argument('--train_db',  required=False)
  parser.add_argument('--test_db',   required=True)
  parser.add_argument('--model_dir', required=True)
  parser.add_argument('--do', nargs='+', choices=['train', 'test', 'eval'], required=True,
            help='perform training, or testing, or both')
  parser.add_argument('--logging_level', default=20, type=int)

  args, args_list_remaining = parser.parse_known_args(arg_list)
  return args, args_list_remaining


def main(args_list):
  args, args_list_remaining = parse_args(arg_list)

  # set GPU
  os.putenv('CUDA_VISIBLE_DEVICES', str(args.gpu))

  # model_dir
  model_dir = args.model_dir

  # test_db_path
  test_db_path = get_test_db_path(args.test_db)

  if 'train' in args.do:

    # train_db_path
    assert args.train_db is not None
    train_db_path = get_train_db_path(args.train_db)

    trainval_net (['--train_db_path', train_db_path,
                   '--val_db_path',   test_db_path,
                   '--model_dir',     model_dir] +
                  list(args_list_remaining))

  # find all models
  logging.info ('Looking for pattern %s' % op.join(model_dir, '*.ckpt.index'))
  model_paths = glob(op.join(model_dir, '*.ckpt.index'))
  logging.info ('Found %d models to test and/or evaluate' % len(model_paths))

  if 'test' in args.do:

    for model_path in sorted(model_paths):
      model_path = model_path[:-6]
      logging.info ('Will test model %s' % model_path)

      out_name = get_detected_name(args.test_db, model_path)
      out_db_path  = op.join(model_dir, 'detected/%s.db' % out_name)
     
      mAP = test_net(['--out_db_path',   out_db_path,
                      '--gt_db_path',    test_db_path,
                      '--model_path',    model_path] +
                      list(args_list))

  if 'eval' in args.do:

    mAPs = {}

    for model_path in sorted(model_paths):
      model_path = model_path[:-6]
      logging.info ('Will evaluate model %s' % model_path)

      out_name = get_detected_name(args.test_db, model_path)
      out_db_path  = op.join(model_dir, 'detected/%s.db' % out_name)
      results_path = op.join(model_dir, 'detected/%s.txt' % out_name) 
      with open(results_path, 'a') as fid:
        fid.write('test_db_path: %s\n' % test_db_path)

      mAPs[model_path] = {}
      for i,car_constraint in enumerate(['width > 25', 'width > 30', 'width > 35']):
        cfg.TEST.CAR_CONSTRAINT = car_constraint
        mAP = reval   (['--out_db_path',   out_db_path,
                        '--results_path',  results_path,
                        '--model_path',    model_path,
                        '--gt_db_path',    test_db_path] +
                       list(args_list))
        mAPs[model_path][car_constraint] = mAP
    pprint.pprint (mAPs)

if __name__ == '__main__':
  arg_list = sys.argv[1:]
  setup_logging(arg_list)
  setup_config(arg_list)
  main(arg_list)
