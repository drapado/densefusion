import os
os.chdir("/home/hoangphuc/DenseFusion")
import torch
import torchfcn
import numpy as np
import copy
from PIL import Image
from torch.autograd import Variable
import pyrealsense2 as rs
import cv2
from pytorchfcn.utils import fcn_deploy
from utils import posenet_deploy
from lib.transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix
import numpy.ma as ma
from lib.network import PoseNet, PoseRefineNet


#eps = 0.000001

cam_cx = 313.425 #325.26110
cam_cy = 328.161 #242.04899
cam_fx = 521.15 #572.41140
cam_fy = 544.577 #573.57043
cam_mat = np.matrix([[cam_fx, 0, cam_cx],[0,cam_fy, cam_cy],[0, 0, 1]])
#distort = [[-0.0475398, 0.399247, 0.0152188, -0.00181894, -0.567924]]
distort = [[0.0, 0.0, 0.0, 0.0, 0.0]]
distort = np.array(distort)

num_objects = 7
objlist = [1, 2, 3, 4, 5, 6, 7]
num_points = 1000
iteration = 4
bs = 1

xmap = np.array([[j for i in range(640)] for j in range(480)])
ymap = np.array([[i for i in range(640)] for j in range(480)])
cam_scale = 1.0

# idx from 0 to 6
idx = 0

# object model 
model_points = posenet_deploy.get_model_points(idx)
model_points = model_points[0].cpu().detach().numpy()


model = "/home/hoangphuc/DenseFusion/trained_models/blisters/pose_model_7_0.01073237437278974.pth"
refine_model = "/home/hoangphuc/DenseFusion/trained_models/blisters/pose_refine_model_88_0.003425211679350735.pth"
estimator = PoseNet(num_points = num_points, num_obj = num_objects)
estimator.cuda()
refiner = PoseRefineNet(num_points = num_points, num_obj = num_objects)
refiner.cuda()
estimator.load_state_dict(torch.load(model))
refiner.load_state_dict(torch.load(refine_model))
estimator.eval()
refiner.eval()



pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
    #video_writer = cv2.VideoWriter("densefusion_realtime.avi", cv2.VideoWriter_fourcc('M','J','P','G'), 6, (640,480))
    while True:
        frames = pipeline.wait_for_frames()
        rgb = frames.get_color_frame()
        rgb = np.asanyarray(rgb.get_data())
        frame = rgb
        
        align = rs.align(rs.stream.color)
        aligned_frames = align.process(frames)
        
        label, label_img = fcn_deploy.fcn_forward(frame)
        cv2.imshow('Mask', label_img)
        cv2.waitKey(1)
        
        if len(np.unique(label)) > 1:
            depth = aligned_frames.get_depth_frame()
            depth = np.asanyarray(depth.get_data())
        
            bbox = list(fcn_deploy.bbox_from_mask(label))
            
            mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
            mask_label = ma.getmaskarray(ma.masked_equal(label, np.array(255)))
            mask = mask_label * mask_depth
            
            rmin, rmax, cmin, cmax = posenet_deploy.get_bbox(bbox)
            
            # choose
            choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
            if len(choose) == 0:
                choose = torch.LongTensor([0])
            if len(choose) > num_points:
                c_mask = np.zeros(len(choose), dtype=int)
                c_mask[:num_points] = 1
                np.random.shuffle(c_mask)
                choose = choose[c_mask.nonzero()]
            else:
                choose = np.pad(choose, (0, num_points - len(choose)), 'wrap')
            
            depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            choose = np.array([choose])
    
            # point cloud
            pt2 = depth_masked / cam_scale
            pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
            pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
            cloud = np.concatenate((pt0, pt1, pt2), axis=1)
            cloud = cloud / 1000.0        
            
            # cropped img
            #img_masked = rgb[:, :, :3]
            img_masked = rgb[:, :, ::-1] # bgr to rgb
            img_masked = np.transpose(img_masked, (2, 0, 1))
            img_masked = img_masked[:, rmin:rmax, cmin:cmax]
            
            # Variables
            cloud = torch.from_numpy(cloud.astype(np.float32)).unsqueeze(0)
            choose = torch.LongTensor(choose.astype(np.int32)).unsqueeze(0)
            img_masked = torch.from_numpy(img_masked.astype(np.float32)).unsqueeze(0)
            index = torch.LongTensor([idx]).unsqueeze(0)  # Specify which object
            
            cloud = Variable(cloud).cuda()
            choose = Variable(choose).cuda()
            img_masked = Variable(img_masked).cuda()
            index = Variable(index).cuda()
            
            # Deploy
            with torch.no_grad():
                pred_r, pred_t, pred_c, emb = estimator(img_masked, cloud, choose, index)
            pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, num_points, 1)
            pred_c = pred_c.view(bs, num_points)
            how_max, which_max = torch.max(pred_c, 1)
            pred_t = pred_t.view(bs * num_points, 1, 3)
            points = cloud.view(bs * num_points, 1, 3)
            
            my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
            my_t = (points.view(bs * num_points, 1, 3) + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
            my_pred = np.append(my_r, my_t)
            
            # Refinement
            for ite in range(0, iteration):
                T = Variable(torch.from_numpy(my_t.astype(np.float32))).cuda().view(1, 3).repeat(num_points, 1).contiguous().view(1, num_points, 3)
                my_mat = quaternion_matrix(my_r)
                R = Variable(torch.from_numpy(my_mat[:3, :3].astype(np.float32))).cuda().view(1, 3, 3)
                my_mat[0:3, 3] = my_t
                
                new_cloud = torch.bmm((cloud - T), R).contiguous()
                pred_r, pred_t = refiner(new_cloud, emb, index)
                pred_r = pred_r.view(1, 1, -1)
                pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))
                my_r_2 = pred_r.view(-1).cpu().data.numpy()
                my_t_2 = pred_t.view(-1).cpu().data.numpy()
                my_mat_2 = quaternion_matrix(my_r_2)
                
                my_mat_2[0:3, 3] = my_t_2
                my_mat_final = np.dot(my_mat, my_mat_2)
                my_r_final = copy.deepcopy(my_mat_final)
                my_r_final[0:3, 3] = 0
                my_r_final = quaternion_from_matrix(my_r_final, True)
                my_t_final = np.array([my_mat_final[0][3], my_mat_final[1][3], my_mat_final[2][3]])
                
                my_pred = np.append(my_r_final, my_t_final)
                my_r = my_r_final
                my_t = my_t_final
                
                print("pred_c_max = " + str(how_max[0])) 
                print("translation vector = " + str(my_t))
                
                my_r_matrix = quaternion_matrix(my_r)[:3, :3]

                projected_points, _ = cv2.projectPoints(model_points, my_r_matrix, my_t, cam_mat, distort)
                projected_points = projected_points.reshape((500,2))
                new_frame = posenet_deploy.drawpointcloud(rgb, projected_points)
                
                #video_writer.write(new_frame)
                cv2.imshow('Result', new_frame)
                cv2.waitKey(1)
            
        
        else:
        # Show images
            #video_writer.write(rgb)
            cv2.imshow('Result', rgb)
            cv2.waitKey(1)
        
    
    
finally:
    # Stop streaming
    pipeline.stop()

