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
import pyrealsense2 as rs
import cv2


model = torchfcn.models.FCN32s(n_class=2)
model.cuda()

checkpoint = torch.load("./trained_models/checkpoint.pth")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

#norm = transforms.Normalize(mean= [112.104004, 115.81309, 114.69222], std= [32.41416, 30.045967, 30.079235])
mean = np.array([112.104004, 115.81309, 114.69222])

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
    #video_writer = cv2.VideoWriter("fcn_realtime.avi", cv2.VideoWriter_fourcc('M','J','P','G'), 6, (640,480))
    while True:
         # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        
        align = rs.align(rs.stream.depth)
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        
        #rgb = np.array(Image.open("/home/hoangphuc/RealSense_Project/rgb_data/181.png"))
        #img = rgb
        
        rgb = color_image[:, :, ::-1].astype(np.float32)
        rgb = color_image - mean
        rgb = np.transpose(rgb, (2, 0, 1))
        #rgb = norm(torch.from_numpy(rgb.astype(np.float32)))
        rgb = torch.from_numpy(rgb.astype(np.float32))
        
        rgb = rgb.reshape(1,3,480,640)
        
        rgb = Variable(rgb).cuda()
        with torch.no_grad():
            semantic = model(rgb)
            
        #out = semantic.data.max(1)[1].cpu().numpy()[:, :, :]
        out = semantic[0].cpu().max(0)[1].data.squeeze(0).byte().numpy()            
        label2img = label2rgb(out, color_image, n_labels=2 )
        #label2img = Image.fromarray(label2img)
        
        #video_writer.write(label2img)
    
             # Show images
        cv2.imshow('color', label2img)
        cv2.waitKey(1)
        
finally:
    # Stop streaming
    pipeline.stop()
