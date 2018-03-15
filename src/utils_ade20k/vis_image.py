import os
from os.path import join
import uuid
import time
import argparse
import pickle
import numpy as np
from imageio import imread, imwrite
import cv2
from collections import OrderedDict

import pycocotools.mask as mask_util

import datasets.dummy_datasets as dummy_datasets
from utils.colormap import colormap
import utils.keypoints as keypoint_utils
from utils.vis import *

import misc as io_utils

def visualize_segmentations(img_path, pkl_path, dataset, images_dir):
    # Open image and pkl
    im = imread(img_path)
    cls_boxes, cls_segms, cls_keyps = pickle.load(open(pkl_path, 'rb'))
    boxes, segms, keypoints, classes = convert_from_cls_format(cls_boxes, cls_segms, cls_keyps)

    masks = mask_util.decode(segms)
    color_list = colormap()
    mask_color_id = 0

    segmentations = []

    scores = boxes[:, -1]
    sorted_inds = np.argsort(scores)
    for i in sorted_inds[::-1]:
        bbox = boxes[i, :4]
        score = boxes[i, -1]
        cls = classes[i]
        class_name = dataset.classes[cls]
        class_str = class_name + ' {:0.2f}'.format(score).lstrip('0')

        vis_im = im.copy()
        # show mask
        if segms is not None and len(segms) > i:
            color_mask = color_list[mask_color_id % len(color_list), 0:3]
            mask_color_id += 1
            vis_im = vis_mask(vis_im, masks[..., i], color_mask, alpha=1.)

        # Crop with context
        x0, y0, x1, y1 = bbox
        bbox_h = y1-y0
        bbox_w = x1-x0
        vis_im = vis_bbox(vis_im, (x0, y0, bbox_w, bbox_h)) # show bbox

        h,w = vis_im.shape[:2]
        y0_c = max(0, int(y0 - .5*bbox_h))
        y1_c = min(h, int(y1 + .5*bbox_h))
        x0_c = max(0, int(x0 - .5*bbox_w))
        x1_c = min(w, int(x1 + .5*bbox_w))

        vis_im = vis_im[y0_c:y1_c,x0_c:x1_c,:]
        im_crop = im[y0_c:y1_c,x0_c:x1_c,:]
        vis_im = np.concatenate((im_crop, vis_im), axis=1)

        # resize
        r = 256./vis_im.shape[0]
        vis_im = cv2.resize(vis_im, dsize=(0,0), fx=r, fy=r, interpolation=cv2.INTER_NEAREST)

        # show class
        vis_im = vis_class(vis_im, (5, 15), class_str)

        vis_path = save(vis_im, images_dir)
        segmentations.append([vis_path, class_name, score])
    return segmentations

def visualize_all_segmentations(img_path, pkl_path, dataset, images_dir, MIN=0, MAX=1):
    # Open image and pkl
    im = imread(img_path)
    cls_boxes, cls_segms, cls_keyps = pickle.load(open(pkl_path, 'rb'))
    boxes, segms, keypoints, classes = convert_from_cls_format(cls_boxes, cls_segms, cls_keyps)

    if boxes is None or boxes.shape[0] == 0:
        return None

    if segms is not None:
        masks = mask_util.decode(segms)
        color_list = colormap()
        mask_color_id = 0

    # Display in largest to smallest order to reduce occlusion
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    sorted_inds = np.argsort(-areas)

    for i in sorted_inds:
        bbox = boxes[i, :4]
        score = boxes[i, -1]
        cls = classes[i]
        if score < MIN or score > MAX:
            continue
        # show bbox
        im = vis_bbox(im, (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]))
        # show class
        class_str = get_class_string(classes[i], score, dataset)
        im = vis_class(im, (bbox[0], bbox[1] - 2), class_str)
        # show mask
        if segms is not None and len(segms) > i:
            color_mask = color_list[mask_color_id % len(color_list), 0:3]
            mask_color_id += 1
            im = vis_mask(im, masks[..., i], color_mask)
        # show keypoints
        if keypoints is not None and len(keypoints) > i:
            im = vis_keypoints(im, keypoints[i], kp_thresh)

    vis_path = save(im, images_dir)
    return vis_path

def save(img, images_dir):
    fname = "{}.jpg".format(uuid.uuid4().hex)
    path = os.path.join(images_dir, fname)
    imwrite(path, img)
    return path

# def add_color(self, img):
#     if img is None:
#         return None, None

#     h,w = img.shape
#     img_color = np.zeros((h,w,3))
#     for i in xrange(1,151):
#         img_color[img == i] = utils.to_color(i)
#     path = self.save(img_color)
#     return img_color, path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', help="Image name")
    parser.add_argument('-pkl', '--pickle', help="Annotations pickle")
    args = parser.parse_args()

    img_path = "/Users/hujh/Documents/UROP_Torralba/datasets/ade20k/images/training/ADE_train_00000001.jpg"
    pkl_path = "/Users/hujh/Documents/UROP_Torralba/datasets/ade20k/predictions/maskrcnn/pkl/training/ADE_train_00000001.pkl"

    dummy_coco_dataset = dummy_datasets.get_coco_dataset()

    images_dir = "tmp/images/"
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    # paths = visualize_segmentations(img_path, pkl_path, dummy_coco_dataset, images_dir)
    paths = visualize_all_segmentations(img_path, pkl_path, dummy_coco_dataset, images_dir)
    print paths

