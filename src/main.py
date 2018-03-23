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

import argparse
import os
import sys
import time
import numpy as np
import pickle

import datasets.dummy_datasets as dummy_datasets
import utils.vis as vis_utils
import my_utils.projects as projects

from model.maskrcnn import MaskRCNN, PRETRAINED
from annotator.sim import SimulatedAnnotator
from dataset_manager import DatasetManager

def main():
    dataset_manager = DatasetManager()
    datasetA = dataset_manager.dataset_nameA
    datasetB = dataset_manager.dataset_nameB
    annotator = SimulatedAnnotator(datasetB)

    initial_weights = PRETRAINED
    model = MaskRCNN(weights=initial_weights)
    if initial_weights is None:
        model.train(datasetA, epoches=10)

    for i in range(10):
        res_file = model.test(datasetB)
        good_annotations = annotator.filter(res_file)
        new_dataset = dataset_manager.create_new_dataset(datasetB, good_annotations)
        # Retrain
        model.train(new_dataset, epoches=2)



        


if __name__ == '__main__':
    main()
