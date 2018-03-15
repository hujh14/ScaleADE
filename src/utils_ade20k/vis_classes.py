import argparse
import os
import random
import uuid
import time
import numpy as np
import pandas as pd

import datasets.dummy_datasets as dummy_datasets
import misc as utils
import vis_image

class Visualizer:

    def __init__(self, project, config, MAX=1000):
        self.project = project
        self.config = config
        self.dataset = dummy_datasets.get_coco_dataset()
        self.MAX = MAX

        self.out_dir = "tmp/{}/classes/".format(self.project)
        self.images_dir = os.path.join(self.out_dir, "images/")
        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir)

        self.print_location(self.out_dir)

        self.out_paths = {}
        self.refresh_rate = 1000

    def init_outfile(self, class_name):
        head = str(self.config)
        html = "<html><head>" + head + "</head><body></body></html>"

        outfile = os.path.join(self.out_dir, "{}.html".format(class_name))
        with open(outfile, 'w') as f:
            f.write(html)
        return outfile

    def print_location(self, loc):
        if 'local' in self.project:
            print loc
        else:
            # Print link to output file
            root = "/data/vision/oliva/scenedataset/"
            abs_path = os.path.abspath(loc)
            rel_path = os.path.relpath(abs_path, root)
            print "http://places.csail.mit.edu/{}".format(rel_path)

    def visualize_im_list(self, im_list):
        for n, line in enumerate(im_list[:self.MAX]):
            im = line.split()[0]
            print n, im
            self.add_paths(im)
            if n == 10 or n % self.refresh_rate == 0:
                self.write()
        self.write()

    def add_paths(self, im):
        img_dir = config["images"]
        pkl_dir = os.path.join(config["predictions"], "maskrcnn/pkl")
        img_path = os.path.join(img_dir, im)
        pkl_path = os.path.join(pkl_dir, im.replace('.jpg', '.pkl'))

        try:
            segs = vis_image.visualize_segmentations(img_path, pkl_path, self.dataset, images_dir=self.images_dir)
            for seg in segs:
                path, class_name, score = seg
                if class_name not in self.out_paths:
                    self.out_paths[class_name] = []
                self.out_paths[class_name].append([path, score])
        except:
            print "Skipping ", im

    def write(self):
        for class_name in self.out_paths:
            outfile = self.init_outfile(class_name)
            paths, scores = zip(*self.out_paths[class_name])
            paths = np.array(paths)
            scores = np.array(scores)
            sorted_idx = np.argsort(scores)[::-1]
            self.write_file(outfile, paths[sorted_idx])

    def write_file(self, outfile, paths):
        image_tags = []
        for path in paths:
            image_tags.append(self.get_image_tag(path, outfile))
        img_section = ' '.join(image_tags)

        # Append to body
        with open(outfile, 'r') as f:
            html = f.read()
        new_html = html.replace("</body>", "{}</body>".format(img_section))
        with open(outfile, 'w') as f:
            f.write(new_html)

    def get_image_tag(self, path, outfile):
        if os.path.isabs(path):
            # Symlink into tmp image directory
            path = self.symlink(path)

        path = os.path.relpath(path, os.path.dirname(outfile))
        return "<img src=\"{}\" height=\"256px\" style=\"image-rendering: pixelated;\">".format(path)

    def symlink(self, path):
        fn = "{}.jpg".format(uuid.uuid4().hex)
        dst = os.path.join(self.images_dir, fn)
        os.symlink(path, dst)
        return dst

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--project', type=str, required=True, help="Project name")
    # parser.add_argument("--prediction", type=str, required=True, help="")
    parser.add_argument('-r', '--randomize', action='store_true', default=False, help="Randomize image list")
    args = parser.parse_args()

    # Configuration
    config = utils.get_config(args.project)
    vis = Visualizer(args.project, config)

    # Image List
    im_list = utils.open_im_list(config["im_list"])

    if args.randomize:
        # Shuffle image list
        random.seed(3)
        random.shuffle(im_list)

    vis.visualize_im_list(im_list)

