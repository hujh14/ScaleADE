from __future__ import print_function
import os
import json
import colorsys
import numpy as np

PATH = os.path.dirname(__file__)

c_to_idx = {}
idx_to_c = {}

def open_category_info():
    with open(os.path.join(PATH, "objectInfo150.txt"), 'r') as f:
        for line in f.readlines():
            split = line.split()
            idx = split[0]
            if idx.isdigit():
                idx = int(idx)
                categories = " ".join(split[4:])
                categories = categories.split(",")
                categories = [c.strip() for c in categories]
                idx_to_c[idx] = categories
                for c in categories:
                    c_to_idx[c] = idx

def category_to_idx(category):
    if category not in c_to_idx:
        return None
    return c_to_idx[category]

def idx_to_category(idx):
    if idx not in idx_to_category:
        return None
    return idx_to_c[idx]
open_category_info()


def open_im_list(im_list_txt):
    if ".txt" not in im_list_txt:
        project = im_list_txt
        CONFIG = get_config(project)
        im_list_txt = CONFIG["im_list"]

    im_list = [line.rstrip() for line in open(im_list_txt, 'r')]
    return np.array(im_list)

def get_config(project):
    with open(os.path.join(PATH, "paths.json"), 'r') as f:
        data_config = json.load(f)
        if project in data_config:
            return data_config[project]
        else:
            raise Exception("Project not found: " + project)