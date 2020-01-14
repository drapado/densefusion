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

model = torchfcn.models.FCN8s(n_class=9)
model.cuda()

checkpoint = torch.load("/home/hoangphuc/DenseFusion/pytorchfcn/trained_models/model_best_64_0.9179305713511433_8s_new_1.pth")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
mean = np.array([85.23675, 87.097015, 83.367805])

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
    semantic = semantic[0].cpu().max(0)[1].data.squeeze(0).byte().numpy()
    label_img = label2rgb(semantic, img, n_labels=9)
    mask = np.where(semantic != 0, 255, semantic)
    return semantic, mask, label_img

def get_class_of_bbox(semantic, bbox):
    semantic_cropped = semantic[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
    unique_vals = np.unique(semantic_cropped)
    
    for class_id in range(1,9):
        if class_id in unique_vals:
            return class_id 

def bbox_from_mask(mask):
    contours, _ = cv2.findContours(mask, 1, 2)
    #(contours, _) = imutils.contours.sort_contours(contours)
    bboxes = []
    
    for cnt in contours:
        if cv2.contourArea(cnt) > 700:
            bboxes.append(tuple(cv2.boundingRect(cnt)))
    return bboxes
