#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 09:45:16 2019

@author: hoangphuc
"""

import os
import torch
import torchfcn
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
import cv2
from pytorchfcn.utils.label2Img import label2rgb

model = torchfcn.models.FCN32s(n_class=2)
model.cuda()

checkpoint = torch.load("/home/hoangphuc/DenseFusion/pytorchfcn/trained_models/checkpoint.pth")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
mean = np.array([112.104004, 115.81309, 114.69222])

def fcn_forward(rgb):
    img = rgb
    #rgb = rgb[:, :, ::-1].astype(np.float32)
    rgb = rgb - mean
    rgb = np.transpose(rgb, (2, 0, 1))
    rgb = torch.from_numpy(rgb.astype(np.float32))
    rgb = rgb.reshape(1,3,480,640)
    rgb = Variable(rgb).cuda()

    with torch.no_grad():
        semantic = model(rgb)
    mask = semantic[0].cpu().max(0)[1].data.squeeze(0).byte().numpy()
    label_img = label2rgb(mask, img, n_labels=2)
    mask = np.where(mask==1, 255, mask)
    return mask, label_img

def bbox_from_mask(mask):
    contour, _ = cv2.findContours(mask, 1, 2)
    x,y,w,h = cv2.boundingRect(contour[0])
    return x,y,w,h