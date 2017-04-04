#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

# Reval = re-eval. Re-evaluate saved detections.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb
from utils.citycam_setup import setup_logging, setup_config
import os, sys, argparse
import numpy as np
import logging
import sqlite3
import pprint


def parse_args(arg_list):
  parser = argparse.ArgumentParser(description='Re-evaluate results')
  parser.add_argument('--gt_db_path',
            help='full path to ground truth .db file',
            required=True)
  parser.add_argument('--out_db_path', required=True,
            help='filepath of output database.')
  parser.add_argument('--results_path',
            help='if specified, results will be appended to that file')
  parser.add_argument('--imdb', dest='imdb_name',
            help='dataset to test',
            default='vehicle', type=str)

  args, _ = parser.parse_known_args(arg_list)
  logging.debug('reval called with args: %s' % args)

  return args


def main(arg_list):
  args = parse_args(arg_list)

  imdb = get_imdb(args.imdb_name, args.gt_db_path)

  conn_out = sqlite3.connect(args.out_db_path)
  c_out = conn_out.cursor()
  logging.info ('Evaluating detections')
  mAP, recalls, precisions = imdb.evaluate_detections(c_det=c_out)
  conn_out.close()

  if args.results_path is not None:
    with open(args.results_path, 'a') as fid:
      fid.write('car_constraint: %s\n' % cfg.TEST.CAR_CONSTRAINT)
      fid.write('%.4f\n' % mAP)
      for i in range(len(recalls)):
        # each recall, precision is a class
        fid.write('%s\n' % recalls[i].tolist())
        fid.write('%s\n' % precisions[i].tolist())

  return mAP


if __name__ == '__main__':
  arg_list = sys.argv[1:]
  setup_logging(arg_list)
  setup_config(arg_list)
  main(arg_list)

