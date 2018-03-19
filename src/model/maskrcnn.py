

"""MaskRCNN model wrapper for active learning."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import cv2
import logging
import numpy as np
import os

from tools.train_net import main as train_main
from tools.infer_dataset import main as infer_main

class MaskRCNN:

    def __init__(self):
        self.config = "configs/e2e_mask_rcnn_R-101-FPN_1x.yaml"
        self.weights = ""

        self.OUTPUT_DIR = "detectron-output/"

    def train(self):
        opts = ["OUTPUT_DIR", self.OUTPUT_DIR]
        args = argparse.Namespace(cfg_file=self.config, opts=opts)
        train_main(args)

    def infer(self, project):
        args = argparse.Namespace(cfg_file=self.config, weights=self.weights, project=project)
        infer_main(args)



if __name__ == '__main__':
    model = MaskRCNN()
    model.train()

