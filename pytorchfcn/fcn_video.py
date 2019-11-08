#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 15:53:05 2019

@author: hoangphuc
"""


import os
import torch
import torchfcn
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
from utils.label2Img import label2rgb
import cv2 
from os.path import isfile, join



import numpy as np


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (
            hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        )
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc


pathIn = "/home/hoangphuc/RealSense_Project/test_label/SegmentationClassPNG"
files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
files.sort()


model = torchfcn.models.FCN32s(n_class=2)
model.cuda()

checkpoint = torch.load("./trained_models/model_best_55_0.9619685916164014.pth")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

#norm = transforms.Normalize(mean= [112.104004, 115.81309, 114.69222], std= [32.41416, 30.045967, 30.079235])
mean = np.array([112.104004, 115.81309, 114.69222])

cap = cv2.VideoCapture("/home/hoangphuc/videofromframes.avi")
i = 0
while cap.isOpened() and i <= len(files):
    
    #rgb = np.array(Image.open("/home/hoangphuc/RealSense_Project/rgb_data/181.png"))
    #img = rgb
    ret, rgb = cap.read()
    img = rgb
    #rgb = rgb[:, :, ::-1].astype(np.float32)
    rgb = rgb - mean
    rgb = np.transpose(rgb, (2, 0, 1))
    #rgb = norm(torch.from_numpy(rgb.astype(np.float32)))
    rgb = torch.from_numpy(rgb.astype(np.float32))
    
    rgb = rgb.reshape(1,3,480,640)
    
    rgb = Variable(rgb).cuda()
    with torch.no_grad():
        semantic = model(rgb)
        print(semantic.size())
        
    #out = semantic.data.max(1)[1].cpu().numpy()[:, :, :]
    out = semantic[0].cpu().max(0)[1].data.squeeze(0).byte().numpy()
    #out = np.reshape(out, (480,640))
    label2img = label2rgb(out, img, n_labels=2 )
    
    filename = pathIn + "/" + files[i]
    target = np.array(Image.open(filename))
    target = torch.from_numpy(target.astype(np.int64))
    target = target.data.numpy()
    acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(target, out, n_class=2)
    print(filename)
    print("Accuracy = " + str(acc))
    print("Accuracy class = " + str(acc_cls))
    print("Mean IU = " + str(mean_iu))
    print("Frequency Weighted AVraged Accuracy = " + str(fwavacc))
    
    cv2.putText(label2img, "IOU = " + str(round(mean_iu,4)), (0,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0))
    
    i += 1
    
    cv2.imshow('color', label2img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break        
