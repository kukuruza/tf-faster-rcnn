import sys, os, os.path as op
sys.path.insert(0, op.join(os.getenv('CITY_PATH'), 'src'))
from learning.dbEvaluate import dbEvalClass
from datasets.imdb import imdb
import numpy as np
import scipy.sparse
import utils.cython_bbox
import logging
import uuid
#from db_eval import eval_class
from model.config import cfg
import sqlite3

class vehicle(imdb):
  ''' Binary classifier vehicles / nonvehicles '''

  def __init__(self, db_path, max_images=None):
    imdb.__init__(self, op.splitext(op.basename(db_path))[0])

    assert op.exists(db_path), 'db_path does not exist: %s' % db_path
    self.conn = sqlite3.connect (db_path)
    self.c    = self.conn.cursor()

    # option to use only a fraction of the dataset
    self.max_images = max_images
    if max_images is not None:
      # TODO: future functinality
      logging.info ('max_images is %s' % str(self.max_images))
    
    self.c.execute('SELECT imagefile FROM images')
    self.imagefiles = self.c.fetchall()
    if self.max_images is not None and self.max_images < len(self.imagefiles):
      np.random.shuffle (self.imagefiles)
      self.imagefiles = self.imagefiles[:self.max_images]

    self._classes = ('__background__', 'vehicle')
    self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
    self._roidb_handler = None
    self._salt = str(uuid.uuid4())
    self._comp_id = 'comp4'

    self.config = {'cleanup'     : True,
                   'use_salt'    : True,
                   'use_diff'    : False,
                   'matlab_eval' : False,
                   'rpn_file'    : None,
                   'min_size'    : 2}


  def num_images(self):
    return len(self.imagefiles)


  def _get_widths(self):
    if self.max_images is None:
      self.c.execute('SELECT width FROM images')
      return [width for (width,) in self.c.fetchall()]
    else: # in case self.max_images is specified, have to go one-by-one
      widths = []
      for imagefile in self.imagefiles:
        self.c.execute('SELECT width FROM images WHERE imagefile=?', imagefile)
        width = self.c.fetchone()[0]
        widths.append(width)

  def gt_roidb(self):
    """
    Return the database of ground-truth regions of interest.
    """
    gt_roidb = []

    for (imagefile,) in self.imagefiles:

      self.c.execute('SELECT x1,y1,width,height FROM cars '
                     'WHERE imagefile=? AND (%s)' % cfg.TRAIN.CAR_CONSTRAINT,
                     (imagefile,))
      entries = self.c.fetchall()
      num_objs = len(entries)
      logging.debug('%d boxes for %s' % (num_objs, imagefile))

      boxes = np.zeros((num_objs, 4), dtype=np.uint16)
      gt_classes = np.zeros((num_objs), dtype=np.int32)
      overlaps = np.zeros((num_objs, 2), dtype=np.float32) # '__background__' & 'vehicle'
      # "Seg" area for pascal is just the box area
      seg_areas = np.zeros((num_objs), dtype=np.float32)
      cls_inds = []

      # Load object bounding boxes into a data frame.
      for ix, (x1,y1,width,height) in enumerate(entries):
          x2 = x1 + width
          y2 = y1 + height
          #cls = self._class_to_ind[obj.find('name').text.lower().strip()]
          cls_inds.append(ix)  # need only our class
          boxes[ix, :] = [x1, y1, x2, y2]
          gt_classes[ix] = 1  # 1 is the 'vehicle' index
          overlaps[ix, 1] = 1.0  # 1 is the 'vehicle' index
          seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

      # need only one class
      boxes = boxes[cls_inds, :]
      gt_classes = gt_classes[cls_inds]
      overlaps = overlaps[cls_inds, :]
      seg_areas = seg_areas[cls_inds]

      overlaps = scipy.sparse.csr_matrix(overlaps)

      self.c.execute('SELECT width,height FROM images WHERE imagefile=?', (imagefile,))
      width,height = self.c.fetchone()

      gt_roidb.append(
             {'imagefile': imagefile,
              'boxes' : boxes,
              'width': width,
              'height': height,
              'gt_classes': gt_classes,
              'gt_overlaps' : overlaps,
              'flipped' : False,
              'seg_areas' : seg_areas})
    return gt_roidb



  def evaluate_detections(self, c_det):
      aps = []
      recalls = []
      precisions = []
      for clsid, cls_name in enumerate(self._classes):
          if cls_name == '__background__':
              continue
          rec, prec, ap = dbEvalClass(c_gt=self.c, c_det=c_det, 
                  params={'ovthresh': 0.5, 'car_constraint': cfg.TEST.CAR_CONSTRAINT})
          #rec, prec, ap = eval_class (self.c, c_det, classname=None, ovthresh=0.5)
          aps += [ap]
          recalls.append(rec)
          precisions.append(prec)
          logging.info('AP for {} = {:.4f}'.format(cls_name, ap))
      logging.info('Mean AP = {:.4f}'.format(np.mean(aps)))
      mAP        = np.around(np.mean(aps), decimals=4)
      recalls    = np.around(recalls, decimals=4)
      precisions = np.around(precisions, decimals=4)
      return mAP, recalls, precisions

