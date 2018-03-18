import os
import argparse
import pickle
import numpy as np
import cv2

import pycocotools.mask as mask_util

import datasets.dummy_datasets as dummy_datasets
import utils.vis as vis_utils
import utils_ade20k.misc as ade20k_utils


def vis(im_name, im, cls_boxes, cls_segms, cls_keyps):
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()
    out_dir = None

    loaded = vis_utils.vis_one_image_opencv(im, cls_boxes, segms=cls_segms, keypoints=cls_keyps, thresh=0.9, kp_thresh=2,
        show_box=False, dataset=None, show_class=False)
    misc.imsave("loaded.png", loaded)

def create_panoptic_segmentation(img, cls_boxes, cls_segms, cls_keyps, thres=0.7):
    boxes, segms, keypoints, classes = vis_utils.convert_from_cls_format(cls_boxes, cls_segms, cls_keyps)
    dataset = dummy_datasets.get_coco_dataset()

    ade_out = np.zeros(img.shape[:2], dtype="uint8")
    coco_out = np.zeros(img.shape[:2], dtype="uint8")
    inst_out = np.zeros(img.shape[:2], dtype="uint8")
    
    if segms is not None:
        masks = mask_util.decode(segms)
        cnt = 1
        # Display in largest to smallest order to reduce occlusion
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        sorted_inds = np.argsort(-areas)
        for i in sorted_inds:
            if boxes[i, -1] < thres: # Score too low
                continue

            mask = masks[...,i]
            mask = np.nonzero(mask)
            class_name = dataset.classes[classes[i]]
            ade_idx = ade20k_utils.category_to_idx(class_name)
            if ade_idx is not None:
                ade_out[mask] = ade_idx
            coco_out[mask] = i
            inst_out[mask] = cnt
            cnt += 1
    out = np.stack([ade_out, coco_out, inst_out], axis=-1)
    return out

def process(img_path, pkl_path, out_path):
    img = cv2.imread(img_path)
    cls_boxes, cls_segms, cls_keyps = pickle.load(open(pkl_path, 'rb'))
    out = create_panoptic_segmentation(img, cls_boxes, cls_segms, cls_keyps)

    if not os.path.isdir(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))
    cv2.imwrite(out_path, out)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--project', type=str, required=True, help="Project name")
    parser.add_argument('-r', '--restart', action='store_true', default=False, help="Restart")
    args = parser.parse_args()

    config = ade20k_utils.get_config(args.project)
    img_dir = config["images"]
    out_dir = os.path.join(config["predictions"], "maskrcnn")
    pkl_dir = os.path.join(out_dir, "pkl")
    panseg_dir = os.path.join(out_dir, "panseg")

    im_list = [line.rstrip() for line in open(config["im_list"], 'r')]

    for i, im_name in enumerate(im_list):
        img_path = os.path.join(img_dir, im_name)

        img_basename = os.path.splitext(im_name)[0]
        pkl_path = os.path.join(pkl_dir, img_basename + '.pkl')
        panseg_path = os.path.join(panseg_dir, img_basename + '.png')

        if os.path.exists(panseg_path) and not args.restart:
            print("Already done")
            continue

        print('Processing {}, {} -> {}'.format(i, pkl_path, panseg_path))
        process(img_path, pkl_path, panseg_path)


if __name__ == '__main__':
    main()
