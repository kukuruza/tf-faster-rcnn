# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Zheqi He, Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.train_val import get_training_roidb, train_net
from model.config import cfg, cfg_from_file, cfg_from_list, get_output_dir, get_output_tb_dir
from datasets.factory import get_imdb
import datasets.imdb
import os.path as op
import argparse
import logging
import pprint
import numpy as np
import sys

import tensorflow as tf
from nets.vgg16 import vgg16
from nets.res101 import Resnet101


def parse_args(arg_list):
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Faster R-CNN network')
  parser.add_argument('--imdb', dest='imdb_name',
            help='dataset to train on',
            default='vehicle', type=str)
  parser.add_argument('--db_path',
            help='full path to .db file',
            required=True)
  parser.add_argument('--db_val_path',
            help='full path to .db file',
            required=True)
  parser.add_argument('--out_model_dir', required=True,
            help='relative to "output"')
  parser.add_argument('--iters', dest='max_iters',
            help='number of iterations to train',
            default=70000, type=int)
  parser.add_argument('--architecture', dest='net',
            choices=['vgg16', 'res101'],
            default='res101', type=str)
  parser.add_argument('--set', dest='set_cfgs',
            help='set config keys', default=None,
            nargs=argparse.REMAINDER)
  parser.add_argument('--logging_level', default=20, type=int)

  args = parser.parse_args(arg_list)
  return args


def combined_roidb(imdb_name, db_path, cache_path):
    def get_roidb(imdb_name, db_path, cache_path):
        imdb = get_imdb(imdb_name, db_path)
        logging.info('Loaded dataset `%s` for training' % imdb.name)
        imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
        logging.info('Set proposal method: %s' % cfg.TRAIN.PROPOSAL_METHOD)
        roidb = get_training_roidb(imdb, cache_path)
        return roidb

    roidb = get_roidb(imdb_name, db_path, cache_path)
    imdb  = get_imdb(imdb_name, db_path)
    return imdb, roidb


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

  np.random.seed(cfg.RNG_SEED)

  # output directory where the models are saved
  output_dir = get_output_dir(args.out_model_dir)
  print('Output will be saved to `{:s}`'.format(output_dir))

  # tensorboard directory where he summaries are saved during training
  tb_dir = get_output_tb_dir(args.out_model_dir)
  print('TensorFlow summaries will be saved to `{:s}`'.format(tb_dir))

  # train set
  imdb, roidb = combined_roidb(args.imdb_name, args.db_path, op.join(output_dir, 'train.pkl'))
  print('{:d} roidb entries'.format(len(roidb)))

  # also add the validation set, but with no flipping images
  _, valroidb = combined_roidb(args.imdb_name, args.db_val_path, op.join(output_dir, 'val.pkl'))
  print('{:d} validation roidb entries'.format(len(valroidb)))

  if args.net == 'vgg16':
    pretrained_model = 'data/imagenet_weights/vgg16.ckpt'
    net = vgg16(batch_size=cfg.TRAIN.IMS_PER_BATCH)
  elif args.net == 'res101':
    pretrained_model = 'data/imagenet_weights/res101.ckpt'
    net = Resnet101(batch_size=cfg.TRAIN.IMS_PER_BATCH)
  else:
    raise NotImplementedError
  assert op.exists(pretrained_model), 'pretrained model does not exist: %s' % pretrained_model
  train_net(net, imdb, roidb, valroidb, output_dir, tb_dir,
            pretrained_model=pretrained_model,
            max_iters=args.max_iters)

if __name__ == '__main__':
  args_list = sys.argv[1:]
  main(args_list)

