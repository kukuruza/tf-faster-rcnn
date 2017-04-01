import logging
import argparse
import tqdm


def print_to_tqdm(t, msg, msg_len=40):
  msg = msg.ljust(msg_len)
  t.set_description(msg)
  t.refresh()


def setup_logging(arg_list):
  '''
  Logging is set up in the "if __main__ " section of each file.
  Logging is prefered over print because it would also output stuff into a file.
  '''
  parser = argparse.ArgumentParser()
  parser.add_argument('--logging_level', default=20, type=int)
  args, arg_list_remaining = parser.parse_known_args(arg_list)

  logging.basicConfig(level=args.logging_level)

  return arg_list_remaining


