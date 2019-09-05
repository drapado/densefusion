#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 10:45:53 2019

@author: hoangphuc
"""

import os
import copy
import random
import argparse
import time
import numpy as np
from PIL import Image
import scipy.io as scio
import scipy.misc
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
from torch.backends import cudnn
from torchvision import transforms
import matplotlib.pyplot as plt

from my_own_data_controller import SegDataset
from loss import Loss
from segnet import SegNet as segnet
import sys
sys.path.append("..")
from lib.utils import setup_logger
from utils.label2Img import label2rgb



model = segnet()
model.cuda()

checkpoint = torch.load("./trained_models/model_22_0.0046774397913876145.pth")
model.load_state_dict(checkpoint)
model.eval()

norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

rgb = np.array(Image.open("/home/hoangphuc/RealSense_Project/rgb_data_done_10/80.png"))
img = rgb
rgb = np.transpose(rgb, (2, 0, 1))
rgb = norm(torch.from_numpy(rgb.astype(np.float32)))
rgb = rgb.reshape(1,3,480,640)

rgb = Variable(rgb).cuda()
with torch.no_grad():
    semantic = model(rgb)
print(semantic[0].shape)

out = semantic[0].cpu().max(0)[1].data.squeeze(0).byte().numpy()
print(out.shape)
#img_out = Image.fromarray(out)

#out = np.array(out)
#out = np.where(out==1, 255, out)


label2img = label2rgb(out, img, n_labels=2 )

img_out = Image.fromarray(label2img)
img_out.show()
