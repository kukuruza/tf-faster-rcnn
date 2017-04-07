from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from citycam import get_test_db_path, main as citycam
from utils.citycam_setup import setup_logging, setup_config
#from model.config import cfg

import logging
from shutil import copyfile
import argparse
import pprint
import sys, os, os.path as op
sys.path.insert(0, op.join(os.getenv('CITY_PATH'), 'src'))
from learning.helperSetup import atcity


def parse_args(arg_list):
  parser = argparse.ArgumentParser(description='Execute train/test pipeline for Citycam.')
  parser.add_argument('--train_db_dir',  required=True,
                      help='E.g. 572-Feb23-09h/jan29-real')
  parser.add_argument('--train_db_names', required=True, nargs='+')
  parser.add_argument('--model_names', required=True, nargs='+')
  parser.add_argument('--test_db',   required=True)

  args, args_list_remaining = parser.parse_known_args(arg_list)
  return args, args_list_remaining


def main(args_list):
  args, args_list_remaining = parse_args(arg_list)
  
  assert len(args.model_names) == len(args.train_db_names)
  for iModel in range(len(args.model_names)):
  
    train_db_name = args.train_db_names[iModel]
    train_db = op.join(args.train_db_dir, train_db_name)

    model_name = args.model_names[iModel]
    model_dir  = op.join('output', op.dirname(args.train_db_dir), model_name)
    if not op.exists(model_dir):
      os.makedirs(model_dir)

    # copy the test db to model_dir (multiprocecessing-safe)
    test_db = get_test_db_path(args.test_db)
    copied_test_db = op.join(model_dir, op.basename(test_db))
    if not op.exists(copied_test_db):
      copyfile(test_db, copied_test_db)
    assert op.isfile(copied_test_db), copied_test_db

    print('job_citycam #%d' % iModel)
    print('  train_db: %s' % train_db)
    print('  copied test_db: %s' % copied_test_db)
    print('  model_dir: %s' % model_dir)

    citycam(['--train_db', train_db,
             '--test_db',  copied_test_db,
             '--model_dir', model_dir] +
            args_list_remaining)

    
if __name__ == '__main__':
  arg_list = sys.argv[1:]
  #arg_list = setup_config(arg_list)
  main(arg_list)

