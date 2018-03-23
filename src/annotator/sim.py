

"""Simulated human annotator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
import pprint
import numpy as np
import os

from pycocotools.cocoeval import COCOeval

from datasets.json_dataset import JsonDataset
import utils.logging

from annotator.annotator_utils import *

PATH = os.path.abspath(os.path.dirname(__file__))
logger = utils.logging.setup_logging(__name__)

class SimulatedAnnotator:

    def __init__(self, dataset_name):
        self.json_dataset = JsonDataset(dataset_name)
        self.output_dir = None

    def filter(self, res_file):
        coco_dt = self.json_dataset.COCO.loadRes(str(res_file))
        coco_eval = COCOeval(self.json_dataset.COCO, coco_dt, 'segm')
        coco_eval.evaluate()

        filtered = []

        results = [x for x in coco_eval.evalImgs if x is not None]
        for result in results:
            imgId = result['image_id']
            catId = result['category_id']
            aRng = result['aRng']
            maxDet = result['maxDet']
            dtIds = result['dtIds']
            dtMatches = result["dtMatches"]
            if len(dtIds) == 0:
                continue
            if coco_eval.params.areaRng[coco_eval.params.areaRngLbl == "all"] != aRng:
                continue

            # print(imgId,catId, aRng,maxDet, dtIds)
            for i, dtId in enumerate(dtIds):
                matches = dtMatches[:,i]
                match = matches[6] # Corresponds to IOU = 0.8
                if match != 0:
                    filtered.append(coco_dt.anns[dtId])

        logger.info("{} annotations -> {} annotations".format(len(coco_dt.getAnnIds()), len(filtered)))
        return filtered

if __name__ == '__main__':
    res_file = "/data/vision/oliva/scenedataset/scaleplaces/ScaleADE/src/model/../workspace/test/ade20k_val/generalized_rcnn/segmentations_ade20k_val_results.json"
    annotator = SimulatedAnnotator("ade20k_val")
    annotator.filter(res_file)

