from __future__ import print_function
import os
import json
import colorsys
import numpy as np

PATH = os.path.abspath(os.path.dirname(__file__))

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