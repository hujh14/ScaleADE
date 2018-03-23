

"""MaskRCNN model wrapper for active learning."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
import numpy as np
import os

from core.config import cfg
from core.config import get_output_dir
from datasets.json_dataset import JsonDataset

from tools.train_net import main as train_net
from tools.test_net import main as test_net
from tools.infer_dataset import main as infer_dataset

PATH = os.path.abspath(os.path.dirname(__file__))
PRETRAINED = "/data/vision/oliva/scenedataset/scaleplaces/ScaleADE/src/model/detectron-output/train/ade20k_train/generalized_rcnn/model_final.pkl"

class MaskRCNN:

    def __init__(self, weights=None):
        self.config = os.path.join(PATH, "configs/e2e_mask_rcnn_R-101-FPN_1x.yaml")
        self.weights = weights

        self.OUTPUT_DIR = os.path.join(PATH, "../workspace/")

    def train(self, dataset, epoches=10):
        opts = ["OUTPUT_DIR", self.OUTPUT_DIR,
                "TRAIN.DATASETS", (dataset,)]
        if self.weights is not None:
            opts.extend(["TRAIN.WEIGHTS", self.weights])

        print("epoches not implemented yet")

        args = argparse.Namespace(cfg_file=self.config, opts=opts)
        train_net(args)

    def test(self, dataset):
        opts = ["OUTPUT_DIR", self.OUTPUT_DIR,
                "TEST.DATASETS", (dataset,)]
        if self.weights is not None:
            opts.extend(["TEST.WEIGHTS", self.weights])

        args = argparse.Namespace(cfg_file=self.config, range=[0,100], opts=opts, multi_gpu_testing=True)
        test_net(args)

        # Get res_file path
        output_dir = get_output_dir(dataset, training=False)
        json_dataset = JsonDataset(dataset)
        res_file = os.path.join(output_dir, 'segmentations_' + json_dataset.name + '_results.json')
        return res_file

    def infer(self, project):
        args = argparse.Namespace(cfg_file=self.config, weights=self.weights, project=project)
        infer_dataset(args)

if __name__ == '__main__':
    model = MaskRCNN(weights=PRETRAINED)
    model.train("ade20k_train")
    # model.test("ade20k_val")

