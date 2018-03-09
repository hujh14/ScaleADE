#!/usr/bin/env python2

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time
import numpy as np
import pickle

from caffe2.python import workspace

from core.config import assert_and_infer_cfg
from core.config import cfg
from core.config import merge_cfg_from_file
from utils.timer import Timer
import core.test_engine as infer_engine
import datasets.dummy_datasets as dummy_datasets
import utils.c2 as c2_utils
import utils.logging
import utils.vis as vis_utils
import utils_ade20k.misc as ade20k_utils

c2_utils.import_detectron_ops()
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default="configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml",
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default="https://s3-us-west-2.amazonaws.com/detectron/35861858/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml.02_32_51.SgT4y1cO/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl",
        type=str
    )
    parser.add_argument('-p',
        '--project',
        help='project_name',
        required=True,
        type=str
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

def main(args):
    logger = logging.getLogger(__name__)
    merge_cfg_from_file(args.cfg)
    cfg.TEST.WEIGHTS = args.weights
    cfg.NUM_GPUS = 1
    assert_and_infer_cfg()
    model = infer_engine.initialize_model_from_cfg()
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()

    config = ade20k_utils.get_config(args.project)
    img_dir = config["images"]
    out_dir = os.path.join(config["predictions"], "maskrcnn")
    pkl_dir = os.path.join(out_dir, "pkl")
    vis_dir = os.path.join(out_dir, "vis")
    
    im_list = [line.rstrip() for line in open(config["im_list"], 'r')]
    im_list = im_list[args.start:args.end]

    for i, im_name in enumerate(im_list):
        img_path = os.path.join(img_dir, im_name)
        img_basename = os.path.splitext(im_name)[0]
        pkl_path = os.path.join(pkl_dir, img_basename + '.pkl')
        vis_path = os.path.join(vis_dir, img_basename + '.png') 
        logger.info('Processing {} -> {}'.format(im_name, vis_path))
        
        if os.path.exists(vis_path):
            print("Already done")
            continue

        im = cv2.imread(img_path)
        timers = defaultdict(Timer)
        t = time.time()
        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                model, im, None, timers=timers
            )
        logger.info('Inference time: {:.3f}s'.format(time.time() - t))
        for k, v in timers.items():
            logger.info(' | {}: {:.3f}s'.format(k, v.average_time))
        if i == 0:
            logger.info(
                ' \ Note: inference on the first image will be slower than the '
                'rest (caches and auto-tuning need to warm up)'
            )

        pkl_obj = (cls_boxes, cls_segms, cls_keyps)
        if not os.path.isdir(os.path.dirname(pkl_path)):
            os.makedirs(os.path.dirname(pkl_path))
        pickle.dump(pkl_obj, open(pkl_path, "wb"))
        
        d, vis_name = os.path.split(vis_path)
        split = os.path.splitext(vis_name)
        vis_name = split[0]
        if not os.path.isdir(d):
            os.makedirs(d)

        vis_utils.vis_one_image(
            im[:, :, ::-1],  # BGR -> RGB for visualization
            vis_name,
            d,
            cls_boxes,
            cls_segms,
            cls_keyps,
            dataset=dummy_coco_dataset,
            box_alpha=0.3,
            show_class=True,
            thresh=0.7,
            kp_thresh=2,
            ext='png'
        )


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    utils.logging.setup_logging(__name__)
    args = parse_args()
    main(args)
