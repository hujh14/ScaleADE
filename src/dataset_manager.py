
"""Manage the dataset paths."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
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

PATH = os.path.abspath(os.path.dirname(__file__))

class DatasetManager:

    def __init__(self, output_dir=None):
        self.output_dir = output_dir
        if self.output_dir is None:
            self.output_dir = "/tmp/ScaleADE/datasets/"

    def random_subset(self, dataset_name, num):
        dataset = DATASETS[dataset_name]
        im_dir = dataset[IM_DIR]
        ann_fn = dataset[ANN_FN]

        ann_file = json.load(open(ann_fn, 'r'))
        images = ann_file["images"]
        annotations = ann_file["annotations"]
        categories = ann_file["categories"]

        new_images = random.sample(images, num)
        ids = set([im['id'] for im in new_images])
        new_annotations = [ann for ann in annotations if ann['image_id'] in ids]

        new_dataset_name = dataset_name + str(uuid.uuid4())
        new_ann_fn = '{}/{}.json'.format("custom", new_dataset_name)
        new_ann_fn = os.path.join(self.output_dir, new_ann_fn)

        self.save_ann_file(new_images, new_annotations, categories, fname=new_ann_fn)
        self.export_to_DATASETS(new_dataset_name, im_dir, new_ann_fn)
        return new_dataset_name

    def split_dataset(self, dataset_name, ratio=0.5):
        dataset = DATASETS[dataset_name]
        im_dir = dataset[IM_DIR]
        ann_fn = dataset[ANN_FN]

        dataset_nameA = dataset_name + '_{}_A'.format(ratio)
        dataset_nameB = dataset_name + '_{}_B'.format(ratio)
        ann_fnA = '{}/{}/{}.json'.format(dataset_name, "splits", dataset_nameA)
        ann_fnB = '{}/{}/{}.json'.format(dataset_name, "splits", dataset_nameB)
        ann_fnA = os.path.join(self.output_dir, ann_fnA)
        ann_fnB = os.path.join(self.output_dir, ann_fnB)

        if not os.path.exists(ann_fnA) or not os.path.exists(ann_fnB):
            ann_file = json.load(open(ann_fn, 'r'))
            images = ann_file["images"]
            annotations = ann_file["annotations"]
            categories = ann_file["categories"]

            n = len(images)
            k = int(ratio*n)
            random.shuffle(images)
            imagesA = images[:k]
            imagesB = images[k:]
            idsA = set([im['id'] for im in imagesA])
            idsB = set([im['id'] for im in imagesB])
            annotationsA = [ann for ann in annotations if ann['image_id'] in idsA]
            annotationsB = [ann for ann in annotations if ann['image_id'] in idsB]

            self.save_ann_file(imagesA, annotationsA, categories, fname=ann_fnA)
            self.save_ann_file(imagesB, annotationsB, categories, fname=ann_fnB)

        self.export_to_DATASETS(dataset_nameA, im_dir, ann_fnA)
        self.export_to_DATASETS(dataset_nameB, im_dir, ann_fnB)
        return dataset_nameA, dataset_nameB

    def create_dataset_with_new_annotations(self, dataset_name, new_annotations):
        dataset = DATASETS[dataset_name]
        old_im_dir = dataset[IM_DIR]
        old_ann_fn = dataset[ANN_FN]
        old_ann_file = json.load(open(old_ann_fn, 'r'))
        old_images = dataset_anns["images"]
        old_annotations = dataset_anns["annotations"]
        old_categories = dataset_anns["categories"]

        # Ground truth must be polygons not masks
        new_annotations = convert_annotations_to_polygon(new_annotations)

        ids = set([ann['image_id'] for ann in new_annotations])
        new_images = [im for im in images if im['id'] in ids]

        # Save
        new_dataset_name = dataset_name + str(uuid.uuid4())
        new_ann_fn = '{}/{}/{}.json'.format(dataset_name, "custom", new_dataset_name)
        new_ann_fn = os.path.join(self.output_dir, new_ann_fn)
        self.save_ann_file(new_images, new_annotations, old_categories, fname=new_ann_fn)
        self.export_to_DATASETS(new_dataset_name, old_im_dir, new_ann_fn)
        return new_dataset_name

    def export_to_DATASETS(self, dataset_name, im_dir, ann_fn):
        # Export to DATASETS variable for use by other programs
        new_dataset = {}
        new_dataset[IM_DIR] = im_dir
        new_dataset[ANN_FN] = ann_fn
        assert os.path.exists(ann_fn)
        DATASETS[dataset_name] = new_dataset
        print("Exported {} dataset.".format(dataset_name))

    def save_ann_file(self, images, annotations, categories, fname=None):
        if fname is None:
            fname = "tmp/{}.json".format(uuid.uuid4())
            os.path.join(self.output_dir, fname)
        if not os.path.isdir(os.path.dirname(fname)):
            os.makedirs(os.path.dirname(fname))

        data_out = {'images': images, 'annotations': annotations, 'categories': categories}
        with open(fname, 'w') as f:
            json.dump(data_out, f)
        print("Created annotations: {}".format(fname))
        return fname

def get_dataset_stats(dataset_name):
    dataset = DATASETS[dataset_name]
    ann_file = json.load(open(dataset[ANN_FN], 'r'))
    images = ann_file["images"]
    annotations = ann_file["annotations"]
    categories = ann_file["categories"]

    stats = {}
    stats["num_images"] = len(images)
    stats["num_annotations"] = len(annotations)
    stats["num_categories"] = len(categories)
    return stats

def convert_annotations_to_polygon(annotations):
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
    return clean_annotations

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


