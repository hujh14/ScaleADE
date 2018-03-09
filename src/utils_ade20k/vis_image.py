import os
from os.path import join
import uuid
import time
import argparse
import numpy as np
from imageio import imread, imwrite
from collections import OrderedDict

import pycocotools.mask as mask_util

from utils.colormap import colormap
import utils.keypoints as keypoint_utils
from utils.vis import *

import misc as io_utils

def get_paths(config, im, images_dir="tmp/images/"):
    img, im_path = get_image(config, im)
    pred_path = config["prediction"]

    # MaskRCNN
    # vis
    vis_dir = join(pred_path, "maskrcnn/vis")
    # pkl
    pkl_dir = join(pred_path, "maskrcnn/pkl")
    pkl_path = join(pkl_dir, im.replace('.jpg', '.pkl'))

    paths = OrderedDict()
    paths["image"] = im_path
    paths["maskrcnn_vis"] = join(vis_dir, im)

def vis_pickle(im, pkl_path, MIN=0, MAX=1):
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

    visualized = False
    for i in sorted_inds:
        bbox = boxes[i, :4]
        score = boxes[i, -1]
        cls = classes[i]
        if score < MIN or score > MAX or cls != show_class:
            continue

        visualized = True
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

    if visualized:
        return im
    return None

def get_image(config, im):
    im_dir = config["images"]
    im_path = os.path.join(im_dir, im)
    img = imread(im_path)
    return img, im_path
    
def save(self, img):
    fname = "{}.jpg".format(uuid.uuid4().hex)
    path = os.path.join(IMAGES_DIR, fname)
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
    parser.add_argument('-p', '--project', required=True, help="Project name")
    parser.add_argument('-i', '--image', help="Image name")
    args = parser.parse_args()

    im = args.image
    if not args.image:
        im_list = io_utils.open_im_list(args.project)
        im = im_list[0]

    print args.project, im
    config = io_utils.get_config(args.project)
    datasource = DataSource(config)
    vis = ImageVisualizer(args.project, datasource)
    paths = vis.visualize(im)
    print paths

