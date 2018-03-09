import argparse
import os
import random
import uuid
import time
import numpy as np
import pandas as pd

import misc as utils
import vis_image

class Visualizer:

    def __init__(self, project, config, MAX=1000):
        self.project = project
        self.config = config
        self.MAX = MAX

        self.out_dir = "tmp/"
        self.out_file = os.path.join(self.out_dir, "{}_{}.html".format(project, int(time.time())))
        self.images_dir = os.path.join(self.out_dir, "images/")
        if not os.path.exists(self.images_dir):
            os.makedirs(images_dir)

        self.init_out_file()

    def init_output_file(self):
        head = str(self.config)
        body = ""
        html = "<html><head>" + head + "</head><body>" + body + "</body></html>"
        with open(self.output_path, 'w') as f:
            f.write(html)

        # Print link to output file
        root = "/data/vision/oliva/scenedataset/"
        abs_path = os.path.abspath(self.output_path)
        rel_path = os.path.relpath(abs_path, root)
        print "http://places.csail.mit.edu/{}".format(rel_path)

    def visualize_images(self, im_list, category=None):
        for n, line in enumerate(im_list[:self.MAX]):
            print n, line
            self.add_image_section(line, category=category)

    def add_image_section(self, line, category=None):
        im = line.split()[0]
        image_tags = []

        paths = vis_image.get_paths(config, im)

        for key in paths:
            tag = self.get_image_tag(paths[key])
            image_tags.append(tag)


        # Build section
        title = "{} {}".format(self.project, line)
        img_section = ' '.join(image_tags)
        section = "<br><br>{}<br><br>{}<br>{}".format(title, img_section)

        # Append to body
        with open(self.output_path, 'r') as f:
            html = f.read()
        new_html = html.replace("</body>", "{}</body>".format(section))
        with open(self.output_path, 'w') as f:
            f.write(new_html)

    # def build_results_section(self, results, order):
    #     keys = []
    #     values = []
    #     for key in results.keys():
    #         keys.append(key)
    #         values.append(results[key])
    #     values = np.stack(values)
        
    #     sorted_values = values[:,order]
    #     df = pd.DataFrame(sorted_values, index=keys, columns=order+1)
    #     html = df.to_html()
    #     return html

    def get_image_tag(self, path):
        if os.path.isabs(path):
            # Symlink into tmp image directory
            path = self.symlink(path)

        path = os.path.relpath(path, os.path.dirname(self.output_path))
        return "<img src=\"{}\" height=\"256px\">".format(path)

    def symlink(self, path):
        fn = "{}.jpg".format(uuid.uuid4().hex)
        dst = os.path.join(IMAGES_DIR, fn)
        os.symlink(path, dst)
        return dst

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--project', type=str, required=True, help="Project name")
    parser.add_argument("--prediction", type=str, required=True, help="")
    parser.add_argument('-r', '--randomize', action='store_true', default=False, help="Randomize image list")
    args = parser.parse_args()

    # Configuration
    config = utils.get_config(args.project)
    if args.prediction is not None:
        config["pspnet_prediction"] = args.prediction

    vis = ProjectVisualizer(config)

    # Image List
    im_list = None
    if args.im_list:
        # Open specific image list
        im_list = utils.open_im_list(args.im_list)
    else:
        # Open default image list
        im_list = utils.open_im_list(config["im_list"])

    if args.randomize:
        # Shuffle image list
        random.seed(3)
        random.shuffle(im_list)

    vis.visualize_images(im_list, category=args.category)

