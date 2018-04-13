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

    epochs = 10
    output_dir = os.path.abspath("workspace")
    model = MaskRCNN(output_dir)

    ratio = 0.2
    datasetA, datasetB = dataset_manager.split_dataset('ade20k_train', ratio=ratio)
    weights = model.train(datasetA, weights=None, epochs=10)
    small_datasetB = dataset_manager.random_subset(datasetB, 1000)
    result = model.predict(small_datasetB, weights)

if __name__ == '__main__':
    main()
