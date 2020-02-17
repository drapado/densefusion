#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 14:36:32 2020

@author: hoangphuc
"""

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import torch
from PIL import Image
from predictor import VisualizationDemo
import pyrealsense2 as rs


image = np.asanyarray(Image.open("/home/hoangphuc/MASK_RCNN_TOMO/input.jpg"))

def setup_realsense():
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    return config


def setup_config():
    model_path = "/home/hoangphuc/MASK_RCNN_TOMO//trained_models/model_final_resnet101_fpn.pth"
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    return cfg

def get_distance_list(boxes, depth_frame):
    dist_list = []
    centers = boxes.get_centers()
    for c in centers:
        dist = depth_frame.get_distance(c[0], c[1])
        dist_list.append(dist)
    return dist_list        
        
    
    


def main():
    cfg = setup_config()
    realsense_cfg = setup_realsense()
    pipeline = rs.pipeline()
    pipeline.start(realsense_cfg)
    visualizer =  VisualizationDemo(cfg)
    
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            color_frame = np.asanyarray(color_frame.get_data())
            frame = color_frame
            
            # Get depth frame
            align = rs.align(rs.stream.color)
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            #depth_frame = np.asanyarray(aligned_depth_frame.get_data())
            
            # Do instance segmentation
            output, vis = visualizer.run_on_image(frame)
            print(output['instances'].pred_classes)
            
            # Calculate distance
            boxes = output['instances'].pred_boxes
            if len(boxes) != 0:
                dist_list = get_distance_list(boxes, aligned_depth_frame)
                for d in dist_list:
                    #cv2.putText(vis, "dist = " + str(d), (50,50), 
                    #            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
                    print(str(dist_list))
            
            cv2.imshow('mask', vis)
            cv2.waitKey(1)
            
        
    finally: 
         # Stop streaming
        pipeline.stop()
            
    
if __name__ == "__main__":
    main()