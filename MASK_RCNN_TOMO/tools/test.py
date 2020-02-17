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

image = np.asanyarray(Image.open("./input.jpg"))

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
model_path = "/home/hoangphuc/MASK_RCNN_TOMO//trained_models/model_final.pth"
cfg.MODEL.WEIGHTS = model_path
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
predictor = DefaultPredictor(cfg)

output = predictor(image)

# We can use `Visualizer` to draw the predictions on the image.
v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
v = v.draw_instance_predictions(output["instances"].to("cpu"))
#cv2.imshow("image", v.get_image()[:, :, ::-1])

result = Image.fromarray(v.get_image()[:, :, ::-1])
result.show()


