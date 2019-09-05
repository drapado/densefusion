#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 09:40:47 2019

@author: hoangphuc
"""

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
import pyrealsense2 as rs
import cv2

norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
model = segnet()
model.cuda()

checkpoint = torch.load("./trained_models/model_22_0.0046774397913876145.pth")
model.load_state_dict(checkpoint)
model.eval()

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
    video_writer = cv2.VideoWriter("realtime_test.avi", cv2.VideoWriter_fourcc('M','J','P','G'), 6, (640,480))
    while True:
         # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        
        rgb = np.transpose(color_image, (2, 0, 1))
        rgb = norm(torch.from_numpy(rgb.astype(np.float32)))
        rgb = rgb.reshape(1,3,480,640)
        rgb = Variable(rgb).cuda()

        with torch.no_grad():
            semantic = model(rgb)
        out = semantic[0].cpu().max(0)[1].data.squeeze(0).byte().numpy()
        label2img = label2rgb(out, color_image, n_labels=2 )
        #img_out = Image.fromarray(label2img)
        video_writer.write(label2img)
        
         # Show images
        cv2.imshow('color', label2img)
        cv2.waitKey(1)

        
        
finally:
    # Stop streaming
    pipeline.stop()
        