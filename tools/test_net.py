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
from utils.loggers import setup_logging
import argparse
import logging
import pprint
import time, os, sys

import tensorflow as tf
from nets.vgg16 import vgg16
from nets.res101 import Resnet101

def parse_args(arg_list):
  parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
  parser.add_argument('--model_path',
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
  parser.add_argument('--results_path',
            help='if specified, results will be appended to that file')
  parser.add_argument('--num_dets', dest='max_per_image',
            help='max number of detections per image',
            default=100, type=int)
  parser.add_argument('--architecture', dest='net',
            choices=['vgg16', 'res101'],
            default='vgg16', type=str)
  parser.add_argument('--set', dest='set_cfgs',
            help='set config keys', default=None,
            nargs=argparse.REMAINDER)

  # parse_known_args since the function can be called from a pipeline
  args, _ = parser.parse_known_args(arg_list)

  print('Called with args:')
  print(args)

  return args


def main(arg_list):
  args = parse_args(arg_list)

  if args.net == 'vgg16':
    cfg_file = 'experiments/cfgs/vgg16.yml'
  elif args.net == 'res101':
    cfg_file = 'experiments/cfgs/res101.yml'
  
  cfg_from_file(cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  #print('Using config:')
  #pprint.pprint(cfg)

  imdb = get_imdb(args.imdb_name, args.gt_db_path)

  tfconfig = tf.ConfigProto(allow_soft_placement=True)
  tfconfig.gpu_options.allow_growth=True

  # reset after potentially previous use of tf
  tf.reset_default_graph()

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

  logging.info(('Loading model check point from {:s}').format(args.model_path))
  saver = tf.train.Saver()
  saver.restore(sess, args.model_path)
  logging.info('Loaded.')

  test_net(sess, net, imdb, args.out_db_path, 
           max_per_image=args.max_per_image,
           results_path=args.results_path)

  sess.close()


if __name__ == '__main__':
  arg_list = sys.argv[1:]
  setup_logging(arg_list)
  main(arg_list)

