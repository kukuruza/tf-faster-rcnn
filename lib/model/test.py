# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
import numpy as np
try:
  import cPickle as pickle
except ImportError:
  import pickle
import sys, os
import math
import sqlite3
from tqdm import trange, tqdm
import logging

from utils.timer import Timer
from utils.cython_nms import nms, nms_new
from utils.boxes_grid import get_boxes_grid
from utils.blob import im_list_to_blob
from utils.loggers import print_to_tqdm

from model.config import cfg
from model.bbox_transform import clip_boxes, bbox_transform_inv

sys.path.insert(0, os.path.join(os.getenv('CITY_PATH'), 'src'))
from learning.helperImg import ReaderVideo
from learning.helperDb import createDb, imageField


def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)

def _get_blobs(im):
  """Convert an image and RoIs within that image into network inputs."""
  blobs = {}
  blobs['data'], im_scale_factors = _get_image_blob(im)

  return blobs, im_scale_factors

def _clip_boxes(boxes, im_shape):
  """Clip boxes to image boundaries."""
  # x1 >= 0
  boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
  # y1 >= 0
  boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
  # x2 < im_shape[1]
  boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
  # y2 < im_shape[0]
  boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
  return boxes

def _rescale_boxes(boxes, inds, scales):
  """Rescale boxes according to image rescaling."""
  for i in range(boxes.shape[0]):
    boxes[i,:] = boxes[i,:] / scales[int(inds[i])]

  return boxes

def im_detect(sess, net, im):
  blobs, im_scales = _get_blobs(im)
  assert len(im_scales) == 1, "Only single-image batch implemented"

  im_blob = blobs['data']
  # seems to have height, width, and image scales
  # still not sure about the scale, maybe full image it is 1.
  blobs['im_info'] = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

  _, scores, bbox_pred, rois = net.test_image(sess, blobs['data'], blobs['im_info'])
  
  boxes = rois[:, 1:5] / im_scales[0]
  # print(scores.shape, bbox_pred.shape, rois.shape, boxes.shape)
  scores = np.reshape(scores, [scores.shape[0], -1])
  bbox_pred = np.reshape(bbox_pred, [bbox_pred.shape[0], -1])
  if cfg.TEST.BBOX_REG:
    # Apply bounding-box regression deltas
    box_deltas = bbox_pred
    pred_boxes = bbox_transform_inv(boxes, box_deltas)
    pred_boxes = _clip_boxes(pred_boxes, im.shape)
  else:
    # Simply repeat the boxes, once for each class
    pred_boxes = np.tile(boxes, (1, scores.shape[1]))

  return scores, pred_boxes

def apply_nms(all_boxes, thresh):
  """Apply non-maximum suppression to all predicted boxes output by the
  test_net method.
  """
  num_classes = len(all_boxes)
  num_images = len(all_boxes[0])
  nms_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
  for cls_ind in range(num_classes):
    for im_ind in range(num_images):
      dets = all_boxes[cls_ind][im_ind]
      if dets == []:
        continue

      x1 = dets[:, 0]
      y1 = dets[:, 1]
      x2 = dets[:, 2]
      y2 = dets[:, 3]
      scores = dets[:, 4]
      inds = np.where((x2 > x1) & (y2 > y1) & (scores > cfg.TEST.DET_THRESHOLD))[0]
      dets = dets[inds,:]
      if dets == []:
        continue

      keep = nms(dets, thresh)
      if len(keep) == 0:
        continue
      nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
  return nms_boxes

def test_net(sess, net, imdb, out_db_path=':memory:', 
             max_per_image=100, thresh=0.05, do_test=False):
  
  np.random.seed(cfg.RNG_SEED)
  """Test a Fast R-CNN network on an image database."""
  num_images = imdb.num_images()
  all_boxes = [[[] for _ in range(num_images)]
         for _ in range(imdb.num_classes)]

  # timers
  _t = {'im_detect' : Timer(), 'misc' : Timer()}

  reader = ReaderVideo()

  if os.path.exists(out_db_path):
    os.remove(out_db_path)
  if not os.path.exists(os.path.dirname(out_db_path)):
    os.makedirs(os.path.dirname(out_db_path))
  conn_out = sqlite3.connect(out_db_path)
  createDb(conn_out)
  c_out = conn_out.cursor()

  t = tqdm(imdb.imagefiles)
  for imid, imagefile in enumerate(t):

    imdb.c.execute('SELECT * FROM images WHERE imagefile=?', imagefile)
    image_entry = imdb.c.fetchone()

    imagefile = imageField(image_entry, 'imagefile')
    s = 'images(imagefile,src,width,height,maskfile,time)'
    c_out.execute('INSERT INTO %s VALUES (?,?,?,?,?,?)' % s, image_entry)

    im = reader.imread(imagefile)

    scores, boxes = im_detect(sess, net, im)

    for clsid in xrange(1, imdb.num_classes):  # clsid = 0 is background

      # apply threshold
      cls_inds = np.where(scores[:, clsid] > thresh)[0]
      cls_scores = scores[cls_inds, clsid]
      cls_boxes = boxes[cls_inds, clsid*4:(clsid+1)*4]
      cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                          .astype(np.float32, copy=False)

      # perform NMS by class
      keep = nms(cls_dets, cfg.TEST.NMS)
      cls_dets = cls_dets[keep, :]

      # save the filtered to the db
      for cls_det in cls_dets:
        score = cls_det.astype(float)[4]
        roi   = cls_det.astype(int)[0:4]
        classname = imdb._classes[clsid]
        s = 'imagefile,name,score,x1,y1,width,height'
        v = (imagefile, classname, score, 
             roi[0], roi[1], roi[2]-roi[0], roi[3]-roi[1])
        c_out.execute('INSERT INTO cars(%s) VALUES (?,?,?,?,?,?,?)' % s, v)

    print_to_tqdm(t, 'Found %d objects' % len(cls_dets))

  conn_out.commit()

  if do_test: 
    logging.info ('Evaluating detections')
    mAP, recalls, precisions = imdb.evaluate_detections(c_det=c_out)

  conn_out.close()
  return

