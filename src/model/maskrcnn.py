

"""MaskRCNN model wrapper for active learning."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os

from core.config import cfg
from datasets import task_evaluation
from datasets.json_dataset import JsonDataset

from tools.train_net import main as train_net
from tools.test_net import main as test_net
from tools.infer_dataset import main as infer_dataset

from dataset_manager import get_dataset_stats

PATH = os.path.abspath(os.path.dirname(__file__))
PRETRAINED = "/data/vision/oliva/scenedataset/scaleplaces/ScaleADE/src/model/detectron-output/train/ade20k_train/generalized_rcnn/model_final.pkl"

class MaskRCNN:

    def __init__(self, output_dir):
        self.config = os.path.join(PATH, "configs/e2e_mask_rcnn_R-101-FPN_1x.yaml")
        self.output_dir = output_dir

    def train(self, dataset, weights=None, epochs=10):
        print("Training: {}, Weights: {}, Epochs: {}".format(dataset, weights, epochs))

        max_iter = epochs * get_dataset_stats(dataset)['num_images']

        # Modify opts
        opts = ["OUTPUT_DIR", self.output_dir,
                "TRAIN.DATASETS", (dataset,),
                "SOLVER.MAX_ITER", max_iter,
                "SOLVER.STEPS", [0, 2./3 * max_iter, 8./9 * max_iter]]
        if weights is not None:
            opts.extend(["TRAIN.WEIGHTS", weights])

        args = argparse.Namespace(cfg_file=self.config, opts=opts, skip_test=True)
        checkpoints = train_net(args)
        return checkpoints["final"]

    def predict(self, dataset, weights):
        print("Testing on {} with weights: {}".format(dataset, weights))

        opts = ["OUTPUT_DIR", self.output_dir,
                "TEST.DATASETS", (dataset,)]
        opts.extend(["TEST.WEIGHTS", weights])

        args = argparse.Namespace(cfg_file=self.config, opts=opts, range=None, multi_gpu_testing=False)
        all_results = test_net(args)

        # Get result file path
        # This was a pretty hacky way to get result file
        output_dir = self.get_output_dir(dataset, training=False)
        res_file = os.path.join(output_dir, 'segmentations_' + dataset_name + '_results.json')
        return res_file

    def get_output_dir(self, dataset_name, training=True):
        # <output-dir>/<train|test>/<dataset-name>/<model-type>/
        tag = 'train' if training else 'test'
        model_type = "generalized_rcnn"
        outdir = os.path.join(self.output_dir, tag, dataset_name, model_type)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        return outdir

if __name__ == '__main__':
    model = MaskRCNN(output_dir="../workspace")
    model.train("ade20k_train", weights=None)

