# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

from datasets.vehicle import vehicle
import numpy as np

def get_imdb(name, db_path, max_images=None):
  if name == 'vehicle':
    return vehicle(db_path, max_images)
  assert False, 'only vehicle dataset supported in this branch'

