
"""Manage the dataset paths."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
import pprint
import cv2
import numpy as np
import os
import json
import random
import uuid

from pycocotools import mask as COCOmask

from datasets.json_dataset import JsonDataset
from datasets.dataset_catalog import *
import utils.logging

PATH = os.path.abspath(os.path.dirname(__file__))
logger = utils.logging.setup_logging(__name__)

class DatasetManager:

    def __init__(self, output_dir=None):
        self.output_dir = output_dir
        if self.output_dir is None:
            self.output_dir = "/tmp/ScaleADE/datasets/"

        A, B = self.split_dataset('ade20k_train', ratio=0.5)
        self.dataset_nameA = A
        self.dataset_nameB = B

    def split_dataset(self, dataset_name, ratio=0.5):
        dataset = DATASETS[dataset_name]
        nameA = dataset_name + 'A'
        nameB = dataset_name + 'B'
        pathA = os.path.join(self.output_dir, '{}/{}/{}.json'.format(dataset_name, ratio, nameA))
        pathB = os.path.join(self.output_dir, '{}/{}/{}.json'.format(dataset_name, ratio, nameB))
        datasetA = {}
        datasetB = {}
        datasetA[IM_DIR] = dataset[IM_DIR]
        datasetB[IM_DIR] = dataset[IM_DIR]
        datasetA[ANN_FN] = pathA
        datasetB[ANN_FN] = pathB
        DATASETS[nameA] = datasetA
        DATASETS[nameB] = datasetB

        if not os.path.exists(pathA) or not os.path.exists(pathB):
            dataset_anns = json.load(open(dataset[ANN_FN], 'r'))
            images = dataset_anns["images"]
            categories = dataset_anns["categories"]
            annotations = dataset_anns["annotations"]

            n = len(images)
            k = int(ratio*n)
            random.shuffle(images)
            imagesA = images[:k]
            imagesB = images[k:]
            idsA = set([im['id'] for im in imagesA])
            idsB = set([im['id'] for im in imagesB])
            annotationsA = [ann for ann in annotations if ann['image_id'] in idsA]
            annotationsB = [ann for ann in annotations if ann['image_id'] in idsB]

            self.output(pathA, imagesA, annotationsA, categories)
            self.output(pathB, imagesB, annotationsB, categories)

        return nameA, nameB

    def output(self, fname, images, annotations, categories):
        logger.info("Writing annotations to " + fname)
        data_out = {'categories': categories, 'images': images, 'annotations': annotations}

        if not os.path.isdir(os.path.dirname(fname)):
            os.makedirs(os.path.dirname(fname))
        with open(fname, 'w') as f:
            json.dump(data_out, f)

    def create_new_dataset(self, dataset_name, new_annotations):
        dataset = DATASETS[dataset_name]
        name = dataset_name + str(uuid.uuid4())
        path = os.path.join(self.output_dir, '{}/{}/{}.json'.format(dataset_name, "random", name))
        new_dataset = {}
        new_dataset[IM_DIR] = dataset[IM_DIR]
        new_dataset[ANN_FN] = path
        DATASETS[name] = new_dataset

        # Write new annotations json
        dataset_anns = json.load(open(dataset[ANN_FN], 'r'))
        images = dataset_anns["images"]
        categories = dataset_anns["categories"]
        annotations = dataset_anns["annotations"]

        clean_annotations = []
        # Convert to polygons
        for ann in new_annotations:
            clean_ann = {k: v for k, v in ann.items()}

            clean_ann['bbox'] = ann['bbox'].tolist()
            clean_ann['area'] = int(ann['area'])
            polygon = rle_to_polygon(ann['segmentation'])
            if polygon is not None:
                clean_ann['segmentation'] = polygon
                clean_annotations.append(clean_ann)

        ids = set([ann['image_id'] for ann in clean_annotations])
        new_images = [im for im in images if im['id'] in ids]

        self.output(path, new_images, clean_annotations, categories)
        return name

def rle_to_polygon(rle):
    mask = COCOmask.decode(rle)
    mask = mask[:,:,np.newaxis]
    mask = np.array(mask, np.uint8)

    mask_new, contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c,h in zip(contours, hierarchy[0]) if h[3] < 0] # Only outermost polygon

    segmentation = []
    for contour in contours:
        polygon = contour.flatten().tolist()
        if len(polygon) >= 6:
            segmentation.append(polygon)
    if len(segmentation) == 0:
        return None

    poly = COCOmask.frPyObjects(segmentation, mask.shape[0], mask.shape[1])[0]
    iou = COCOmask.iou([rle], [poly], [0])[0,0]

    # BUGS ARE PRESENT!!
    # if iou < 0.9:
    #     poly = COCOmask.decode(poly)
    #     rle = COCOmask.decode(rle)
    #     fname = "/tmp/blah/" + str(uuid.uuid4()) + ".png"
    #     cv2.imwrite(fname, np.array(poly != rle, dtype=np.uint8)*255)
    #     print(iou, fname)
    return segmentation

if __name__ == '__main__':
    dataset_manager = DatasetManager()
    dname = "ade20k_val"

    from annotator.sim import SimulatedAnnotator
    res_file = "/data/vision/oliva/scenedataset/scaleplaces/ScaleADE/src/model/../workspace/test/ade20k_val/generalized_rcnn/segmentations_ade20k_val_results.json"
    annotator = SimulatedAnnotator(dname)
    annotations = annotator.filter(res_file)

    dataset_manager.create_new_dataset(dname, annotations)


