# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Zheqi he, Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.test import test_net
from model.config import cfg, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb
import argparse
import logging
import pprint
import time, os, sys

import tensorflow as tf
from nets.vgg16 import vgg16
from nets.res101 import Resnet101

def parse_args(arg_list):
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
  parser.add_argument('--in_model_file',
            help='trained model weights',
            required=True, type=str)
  parser.add_argument('--imdb', dest='imdb_name',
            help='dataset to test',
            default='vehicle', type=str)
  parser.add_argument('--gt_db_path',
            help='full path to ground truth .db file',
            required=True)
  parser.add_argument('--out_db_path', default=':memory:',
            help='filepath of output database. Default is in-memory')
  #parser.add_argument('--comp', dest='comp_mode', help='competition mode',
  #          action='store_true')
  parser.add_argument('--num_dets', dest='max_per_image',
            help='max number of detections per image',
            default=100, type=int)
  parser.add_argument('--architecture', dest='net',
            choices=['vgg16', 'res101'],
            default='res101', type=str)
  parser.add_argument('--set', dest='set_cfgs',
            help='set config keys', default=None,
            nargs=argparse.REMAINDER)
  parser.add_argument('--logging_level', default=20, type=int)

  args = parser.parse_args(arg_list)
  return args

def main(args_list):
  args = parse_args(args_list)
  logging.basicConfig(level=args.logging_level)

  print('Called with args:')
  print(args)

  if args.net == 'vgg16':
    cfg_file = 'experiments/cfgs/vgg16.yml'
  elif args.net == 'res101':
    cfg_file = 'experiments/cfgs/res101.yml'
  
  cfg_from_file(cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)

  imdb = get_imdb(args.imdb_name, args.gt_db_path)
  #imdb.competition_mode(args.comp_mode)

  tfconfig = tf.ConfigProto(allow_soft_placement=True)
  tfconfig.gpu_options.allow_growth=True

  # init session
  sess = tf.Session(config=tfconfig)
  # load network
  if args.net == 'vgg16':
    net = vgg16(batch_size=1)
  elif args.net == 'res101':
    net = Resnet101(batch_size=1)
  else:
    raise NotImplementedError
  anchors = [4, 8, 16, 32]

  net.create_architecture(sess, "TEST", imdb.num_classes,  
                          tag='default', anchor_scales=anchors)

  print(('Loading model check point from {:s}').format(args.in_model_file))
  saver = tf.train.Saver()
  saver.restore(sess, args.in_model_file)
  print('Loaded.')

  test_net(sess, net, imdb, args.out_db_path, max_per_image=args.max_per_image)

  sess.close()


if __name__ == '__main__':
  args_list = sys.argv[1:]
  main(args_list)

