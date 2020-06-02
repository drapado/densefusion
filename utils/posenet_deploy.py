import os
os.chdir("/home/tomo/densefusion")
import random
import numpy as np
import yaml
import copy
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from lib.network import PoseNet, PoseRefineNet
from lib.loss import Loss
from lib.loss_refiner import Loss_refine
from lib.transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix
from lib.knn.__init__ import KNearestNeighbor
import cv2
from PIL import Image, ImageDraw
import numpy.ma as ma
#from pytorchfcn.utils.fcn_deploy import bbox_from_mask 
from scipy.spatial.distance import euclidean



#border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]

# NEW_CAMERA 1280x720
# cam_cx = 665.29
# cam_cy = 349.38
# cam_fx = 925.89
# cam_fy = 925.89 * 0.99821077
# cam_mat = np.matrix([[cam_fx, 0, cam_cx],[0,cam_fy, cam_cy],[0, 0, 1]])
# distort = [[0.164599153, -0.547279413, -0.000301065, -0.000823868, 0.52405535]]
# distort = np.array(distort)

# Avarage cam 1
cam_cx = (981.4441183 + 12)/1.5
cam_cy = (529.055484 - 11.2)/1.5
cam_fx = (1392.240922 - 10.3)/1.5
cam_fy = (1392.240922 - 10.3)/1.5 * 0.99821077
cam_mat = np.matrix([[cam_fx, 0, cam_cx],[0,cam_fy, cam_cy],[0, 0, 1]])
distort = [[0.164599153,	-0.547279413,	-0.000301065,	-0.000823868,	0.52405535]]
distort = np.array(distort)

# cam_cx = (984.5737679)/1.5
# cam_cy = (535.0715399)/1.5
# cam_fx = (1388.835364)/1.5
# cam_fy = (1388.835364)/1.5 * 0.997798539
# cam_mat = np.matrix([[cam_fx, 0, cam_cx],[0,cam_fy, cam_cy],[0, 0, 1]])
# distort = [[0.162875304,	-0.546295228,	0.00002908,	-0.000776614,	0.507221168]]
# distort = np.array(distort)

# cam_cx = 317.718 #652.28949 
# cam_cy = 241.809 #353.75137 
# cam_fx = 615.601 #929.53723 
# cam_fy = 615.896 #929.63879
# cam_mat = np.matrix([[cam_fx, 0, cam_cx],[0,cam_fy, cam_cy],[0, 0, 1]])
# distort = [[0.162875304,	-0.546295228,	0.00002908,	-0.000776614,	0.507221168]]
# distort = np.array(distort)

num_points = 500
cam_scale = 1.0


""" border_list = list(range(0, 641, 1))
border_list[0] = -1

img_width = 480 
img_length = 640

def get_bbox(bbox):
    bbx = [bbox[1], bbox[1] + bbox[3], bbox[0], bbox[0] + bbox[2]]
    if bbx[0] < 0:
        bbx[0] = 0
    if bbx[1] >= 480:
        bbx[1] = 479
    if bbx[2] < 0:
        bbx[2] = 0
    if bbx[3] >= 640:
        bbx[3] = 639                
    rmin, rmax, cmin, cmax = bbx[0], bbx[1], bbx[2], bbx[3]
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > 480:
        delt = rmax - 480
        rmax = 480
        rmin -= delt
    if cmax > 640:
        delt = cmax - 640
        cmax = 640
        cmin -= delt
    return rmin, rmax, cmin, cmax  """
 
 
 #----------------------------- FOR 1280x720 ----------------------------------

border_list = list(range(0, 1281, 1))
border_list[0] = -1

img_width = 720
img_length = 1280

def get_bbox(bbox):
    bbx = [bbox[1], bbox[1] + bbox[3], bbox[0], bbox[0] + bbox[2]]
    if bbx[0] < 0:
        bbx[0] = 0
    if bbx[1] >= 720:
        bbx[1] = 719
    if bbx[2] < 0:
        bbx[2] = 0
    if bbx[3] >= 1280:
        bbx[3] = 1279                
    rmin, rmax, cmin, cmax = bbx[0], bbx[1], bbx[2], bbx[3]
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > 720:
        delt = rmax - 720
        rmax = 720
        rmin -= delt
    if cmax > 1280:
        delt = cmax - 1280
        cmax = 1280
        cmin -= delt
    return rmin, rmax, cmin, cmax  


""" #----------------------------- FOR 1920x1080 ----------------------------------

#border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
border_list = list(range(0, 1961, 40))
border_list[0] = -1

img_width = 1080
img_length = 1920

def get_bbox(bbox):
    bbx = [bbox[1], bbox[1] + bbox[3], bbox[0], bbox[0] + bbox[2]]
    if bbx[0] < 0:
        bbx[0] = 0
    if bbx[1] >= 1080:
        bbx[1] = 1079
    if bbx[2] < 0:
        bbx[2] = 0
    if bbx[3] >= 1920:
        bbx[3] = 1919                
    rmin, rmax, cmin, cmax = bbx[0], bbx[1], bbx[2], bbx[3]
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > 1080:
        delt = rmax - 1080
        rmax = 1080
        rmin -= delt
    if cmax > 1920:
        delt = cmax - 1920
        cmax = 1920
        cmin -= delt
    return rmin, rmax, cmin, cmax
 """



def ply_vtx(path):
    f = open(path)
    assert f.readline().strip() == "ply"
    f.readline()
    f.readline()
    #f.readline()
    N = int(f.readline().split()[-1])
    while f.readline().strip() != "end_header":
        continue
    pts = []
    for _ in range(N):
        pts.append(np.float32(f.readline().split()[:3]))
    return np.array(pts)

def drawpointcloud(image, projected_points):
    frame = Image.fromarray(image)
    draw = ImageDraw.Draw(frame)
    draw.point(projected_points)
    return np.array(frame)



def get_model_points(idx):
    #dataset_root = "/home/tomo/densefusion/datasets/blister_models"
    dataset_root = "/home/tomo/TOMO_BLISTERS_3D"
    model_path = dataset_root + "/models/obj_{:01}/obj_{:01}.ply"
    pt = ply_vtx(model_path.format(idx+1, idx+1))
    model_points = pt / 1000.0
    
    dellist = [j for j in range(0, len(model_points))]
    dellist = random.sample(dellist, len(model_points) - num_points)
    model_points = np.delete(model_points, dellist, axis=0)
    model_points = torch.from_numpy(model_points.astype(np.float32)).unsqueeze(0)
    model_points = Variable(model_points).cuda()
    
    return model_points

def draw_axes(image, my_r_matrix, my_t):
    frame = image
    points = np.float32([[0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.05], [0, 0, 0]]).reshape(-1, 3)
    axisPoints, _ = cv2.projectPoints(points, my_r_matrix, my_t, cam_mat, distort)
    axisPoints.reshape((4,2))
    cv2.line(frame, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), (0,0,255), 3)
    cv2.line(frame, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), (0,255,0), 3)
    cv2.line(frame, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), (255,0,0), 3)
    return frame

""" def find_furthest_dist_on_y(model_points, my_r_matrix, my_t):
    #min_vals_axes = np.amin(model_points, axis=0)
    #max_vals_axes = np.amax(model_points, axis=0)
    
    min_index_axes = np.argmin(model_points, axis=0)
    max_index_axes = np.argmax(model_points, axis=0)
    
    #point_1 = [0, min_vals_axes[0], 0]
    point_1 = model_points[min_index_axes[2]]
    #point_1 = np.array(point_1)
    
    #point_2 = [0, max_vals_axes[0], 0]
    point_2 = model_points[max_index_axes[2]]
    #point_2 = np.array(point_2)
    
    points = [point_1, point_2]
    points = np.array(points)
    
    plane_points, _ = cv2.projectPoints(points, my_r_matrix, my_t, cam_mat, distort)
    plane_points.reshape((2,2))
    
    return plane_points[0], plane_points[1] """

def get_3d_bbox(frame, model_points, my_r_matrix, my_t):
    min_values_axes = np.min(model_points, axis=0)
    max_values_axes = np.max(model_points, axis=0)
    
    #front side
    corner_1 = [max_values_axes[0], max_values_axes[1], max_values_axes[2]]
    corner_2 = [min_values_axes[0], max_values_axes[1], max_values_axes[2]]
    corner_3 = [min_values_axes[0], min_values_axes[1], max_values_axes[2]]
    corner_4 = [max_values_axes[0], min_values_axes[1], max_values_axes[2]]
    
    # back side
    corner_5 = [max_values_axes[0], max_values_axes[1], min_values_axes[2]]
    corner_6 = [min_values_axes[0], max_values_axes[1], min_values_axes[2]]
    corner_7 = [min_values_axes[0], min_values_axes[1], min_values_axes[2]]
    corner_8 = [max_values_axes[0], min_values_axes[1], min_values_axes[2]]
    
    bbox_points = np.array([corner_1, corner_2, corner_3, corner_4,
          corner_5, corner_6, corner_7, corner_8]).reshape(-1,3)
    bbox_points, _ = cv2.projectPoints(bbox_points, my_r_matrix, my_t, cam_mat, distort)
    bbox_points = bbox_points.astype(int)
    bbox_points.reshape((8,2))
    
    #cv2.rectangle(frame, tuple(bbox_points[1].ravel()), tuple(bbox_points[3].ravel()), (0,0,255), 2)
    #cv2.rectangle(frame, tuple(bbox_points[5].ravel()), tuple(bbox_points[7].ravel()), (0,0,255), 2)
    #cv2.rectangle(frame, tuple(bbox_points[0].ravel()), tuple(bbox_points[7].ravel()), (0,0,255), 2)
    #cv2.rectangle(frame, tuple(bbox_points[1].ravel()), tuple(bbox_points[6].ravel()), (0,0,255), 2)
    #cv2.rectangle(frame, tuple(bbox_points[5].ravel()), tuple(bbox_points[0].ravel()), (0,0,255), 2)
    #cv2.rectangle(frame, tuple(bbox_points[6].ravel()), tuple(bbox_points[3].ravel()), (0,0,255), 2)
    
    cv2.line(frame, tuple(bbox_points[0].ravel()), tuple(bbox_points[1].ravel()), (0,0,255), 2)
    cv2.line(frame, tuple(bbox_points[0].ravel()), tuple(bbox_points[3].ravel()), (0,0,255), 2)
    cv2.line(frame, tuple(bbox_points[0].ravel()), tuple(bbox_points[4].ravel()), (0,0,255), 2)
    cv2.line(frame, tuple(bbox_points[1].ravel()), tuple(bbox_points[2].ravel()), (0,0,255), 2)
    cv2.line(frame, tuple(bbox_points[1].ravel()), tuple(bbox_points[5].ravel()), (0,0,255), 2)
    cv2.line(frame, tuple(bbox_points[2].ravel()), tuple(bbox_points[3].ravel()), (0,0,255), 2)
    cv2.line(frame, tuple(bbox_points[2].ravel()), tuple(bbox_points[6].ravel()), (0,0,255), 2)
    cv2.line(frame, tuple(bbox_points[3].ravel()), tuple(bbox_points[7].ravel()), (0,0,255), 2)
    cv2.line(frame, tuple(bbox_points[4].ravel()), tuple(bbox_points[5].ravel()), (0,0,255), 2)
    cv2.line(frame, tuple(bbox_points[4].ravel()), tuple(bbox_points[7].ravel()), (0,0,255), 2)
    cv2.line(frame, tuple(bbox_points[5].ravel()), tuple(bbox_points[6].ravel()), (0,0,255), 2)
    cv2.line(frame, tuple(bbox_points[6].ravel()), tuple(bbox_points[7].ravel()), (0,0,255), 2)
    
    
    #cv2.circle(frame, tuple(bbox_points[0].ravel()), 3, (0,0,255), 1)
    #cv2.circle(frame, tuple(bbox_points[1].ravel()), 3, (0,0,255), 1)
    #cv2.circle(frame, tuple(bbox_points[2].ravel()), 3, (0,0,255), 1)
    #cv2.circle(frame, tuple(bbox_points[3].ravel()), 3, (0,0,255), 1)
    #cv2.circle(frame, tuple(bbox_points[4].ravel()), 3, (0,0,255), 1)
    #cv2.circle(frame, tuple(bbox_points[5].ravel()), 3, (0,0,255), 1)
    #cv2.circle(frame, tuple(bbox_points[6].ravel()), 3, (0,0,255), 1)
    #cv2.circle(frame, tuple(bbox_points[7].ravel()), 3, (0,0,255), 1)
    
    return frame

def putText(frame, i, num_idx, class_idx, translation):
    pos_0 = (20, 50 + i * 50)
    pos_1 = (50, 50 + i * 50) 
    pos_2 = (200, 50 + i * 50)
    
    cv2.putText(frame, str(num_idx), pos_0, cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0,255,0), 2, cv2.LINE_AA)
    cv2.putText(frame, "Class idx: " + str(class_idx), pos_1, cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0,255,0), 2, cv2.LINE_AA)
    cv2.putText(frame, "XYZ: " + str(translation), pos_2, cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0,255,0), 2, cv2.LINE_AA)
