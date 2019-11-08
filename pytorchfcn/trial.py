#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 13:28:45 2019

@author: hoangphuc
"""

import numpy as np
import os.path as osp

import numpy as np
import PIL.Image
import scipy.io

dataset_dir = osp.join("/home/hoangphuc/pytorch-fcn", 'benchmark/benchmark_RELEASE/dataset')
 
label_file = osp.join(dataset_dir, "cls/2008_000002.mat")

mat = scipy.io.loadmat(label_file)
mask = mat['GTcls'][0]['Segmentation'][0]

print(np.unique(mask))