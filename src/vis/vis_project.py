import argparse
import os
import random
import uuid
import time
import numpy as np
import pandas as pd

import datasets.dummy_datasets as dummy_datasets
import my_utils.projects as projects

import vis_image

class Visualizer:

    def __init__(self, project, config, MAX=1000, dataset_name="ade"):
        self.project = project
        self.config = config
        self.MAX = MAX

        if dataset_name == "ade":
            self.dataset = dummy_datasets.get_ade_dataset()
        else:
            self.dataset = dummy_datasets.get_coco_dataset()

        self.out_dir = "tmp/html/"
        self.images_dir = os.path.join(self.out_dir, "images/")
        self.outfile = os.path.join(self.out_dir, "{}_{}.html".format(project, int(time.time())))
        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir)

        self.init_outfile()

    def init_outfile(self):
        head = str(self.config)
        body = ""
        html = "<html><head>" + head + "</head><body>" + body + "</body></html>"
        with open(self.outfile, 'w') as f:
            f.write(html)

        if 'local' in self.project:
            print self.outfile
        else:
            # Print link to output file
            root = "/data/vision/oliva/scenedataset/"
            abs_path = os.path.abspath(self.outfile)
            rel_path = os.path.relpath(abs_path, root)
            print "http://places.csail.mit.edu/{}".format(rel_path)

    def visualize_images(self, im_list):
        for n, line in enumerate(im_list[:self.MAX]):
            print n, line
            self.add_image_section(line)

    def add_image_section(self, line):
        im = line.split()[0]
        image_tags = []

        img_dir = config["images"]
        pkl_dir = os.path.join(config["predictions"], "maskrcnn/pkl")
        img_path = os.path.join(img_dir, im)
        pkl_path = os.path.join(pkl_dir, im.replace('.jpg', '.pkl'))

        all_segs = vis_image.visualize_all_segmentations(img_path, pkl_path, self.dataset, images_dir=self.images_dir)
        image_tags.append(self.get_image_tag(all_segs))

        segs = vis_image.visualize_segmentations(img_path, pkl_path, self.dataset, images_dir=self.images_dir)
        for seg in segs:
            path = seg[0]
            tag = self.get_image_tag(path)
            image_tags.append(tag)


        # Build section
        title = "{} {}".format(self.project, line)
        img_section = ' '.join(image_tags)
        section = "<br><br>{}<br><br>{}".format(title, img_section)

        # Append to body
        with open(self.outfile, 'r') as f:
            html = f.read()
        new_html = html.replace("</body>", "{}</body>".format(section))
        with open(self.outfile, 'w') as f:
            f.write(new_html)

    def get_image_tag(self, path):
        if os.path.isabs(path):
            # Symlink into tmp image directory
            path = self.symlink(path)

        path = os.path.relpath(path, os.path.dirname(self.outfile))
        return "<img src=\"{}\" height=\"256px\" style=\"image-rendering: pixelated;\">".format(path)

    def symlink(self, path):
        fn = "{}.jpg".format(uuid.uuid4().hex)
        dst = os.path.join(self.images_dir, fn)
        os.symlink(path, dst)
        return dst

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--project', type=str, required=True, help="Project name")
    parser.add_argument('-d', '--dataset', type=str, default="ade", help="Dataset: ade, coco, etc")
    parser.add_argument('-r', '--randomize', action='store_true', default=False, help="Randomize image list")
    args = parser.parse_args()

    # Configuration
    config = projects.get_config(args.project)
    vis = Visualizer(args.project, config, dataset_name=args.dataset)

    # Image List
    im_list = projects.open_im_list(config["im_list"])

    if args.randomize:
        # Shuffle image list
        random.seed(3)
        random.shuffle(im_list)

    vis.visualize_images(im_list)

