import logging
import argparse
import tqdm
import os, os.path as op
from model.config import get_experiment


def print_to_tqdm(t, msg, msg_len=40):
  msg = msg.ljust(msg_len)
  t.set_description(msg)
  t.refresh()


def setup_logging(arg_list):
  '''
  Logging is set up in the "if __main__ " section of each file.
  Logging is prefered over print because it would also output stuff into a file.
  '''
  # get logging level, and logfile name
  parser = argparse.ArgumentParser('get logging_level and return the rest')
  parser.add_argument('--logging_level', default=20, type=int)
  parser.add_argument('--train_db_file', required=True)
  args, _ = parser.parse_known_args(arg_list)

  experiment = get_experiment(args.train_db_file)
  log_path = op.join('output', experiment, 'log.txt')

  # init the main logger
  logger = logging.getLogger()
  logger.setLevel(level=args.logging_level)

  # we will write from INFO and up to screen
  ch = logging.StreamHandler()
  ch.setLevel(logging.INFO)
  ch.setFormatter(logging.Formatter("%(message)s"))
  logger.addHandler(ch)

  # we will write everything from DEBUG up to a file
  ch = logging.FileHandler(log_path)
  ch.setLevel(logging.DEBUG)
  ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
  logger.addHandler(ch)
  
  logging.info('log file is %s' % log_path)

