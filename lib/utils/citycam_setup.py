import logging
import argparse
import tqdm
import os, os.path as op
import pprint
from model.config import cfg, cfg_from_file, cfg_from_list


def print_to_tqdm(t, msg, msg_len=50):
  msg = msg.ljust(msg_len)
  t.set_description(msg)
  t.refresh()


def setup_config(arg_list):
  '''
  Read cfg from file and command line
  '''
  parser = argparse.ArgumentParser(add_help=False)
  parser.add_argument('--architecture', dest='net',
            choices=['vgg16', 'res101'],
            default='vgg16', type=str)
  parser.add_argument('--set', dest='set_cfgs',
            help='set config keys', default=None,
            nargs=argparse.REMAINDER)
  args, args_list_remaining = parser.parse_known_args(arg_list)

  if args.net == 'vgg16':
    cfg_file = 'experiments/cfgs/vgg16.yml'
  elif args.net == 'res101':
    cfg_file = 'experiments/cfgs/res101.yml'

  cfg_from_file(cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  logging.debug(pprint.pformat(cfg))

  return args_list_remaining + ['--architecture', args.net]


def setup_logging(arg_list):
  '''
  Logging is set up in the "if __main__ " section of each file.
  Logging is prefered over print because it would also output stuff into a file.
  '''
  # get logging level, and logfile name
  parser = argparse.ArgumentParser(add_help=False)
  parser.add_argument('--logging_level', default=20, type=int)
  parser.add_argument('--model_dir', required=True)
  args, _ = parser.parse_known_args(arg_list)

  if not op.isdir(args.model_dir):
    os.makedirs(args.model_dir)

  # get the main logger
  logger = logging.getLogger()
  logger.setLevel(level=logging.DEBUG)

  logging.debug('had %d logger handlers' % len(logger.handlers))
  for handler in logger.handlers:

    if isinstance(handler, logging.FileHandler):
      handler.stream.close()
      logger.removeHandler(handler)

    if isinstance(handler, logging.StreamHandler):
      handler.setLevel(level=args.logging_level)
      handler.setFormatter(logging.Formatter("%(message)s"))
      #logger.addHandler(ch)

  # now FileHandlers are removed, and StreamHandlers are configured. Add a new FileHandler
  log_path = op.join(args.model_dir, 'log.txt')
  ch = logging.FileHandler(log_path)
  # we will write everything from DEBUG up to a file
  ch.setLevel(logging.DEBUG)
  ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
  logger.addHandler(ch)
  
  logging.debug('now have %d logger handlers' % len(logger.handlers))
  logging.info('log file is %s' % log_path)

