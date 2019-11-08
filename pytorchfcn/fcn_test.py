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



model = torchfcn.models.FCN32s(n_class=2)
model.cuda()

checkpoint = torch.load("./trained_models/checkpoint.pth")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

#norm = transforms.Normalize(mean= [112.104004, 115.81309, 114.69222], std= [32.41416, 30.045967, 30.079235])
mean = np.array([112.104004, 115.81309, 114.69222])

rgb = np.array(Image.open("/home/hoangphuc/DenseFusion_TestData/data_7/11_rgb.png"))
img = rgb

rgb = rgb[:, :, ::-1].astype(np.float32)
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
print(out.shape)

label2img = label2rgb(out, img, n_labels=2 )

label2img = Image.fromarray(label2img)
label2img.show()
out = np.where(out==1, 255, out)
out = Image.fromarray(out)
out.show()