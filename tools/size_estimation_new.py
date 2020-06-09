import os
os.chdir("/home/tomo/densefusion")
import torch
import numpy as np
import copy
from PIL import Image
from torch.autograd import Variable
import pyrealsense2 as rs
import cv2
from utils import posenet_deploy
from lib.transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix
import numpy.ma as ma
from lib.network import PoseNet, PoseRefineNet
from scipy.spatial.distance import euclidean
from MASK_RCNN_TOMO.tools import predictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2 import model_zoo
from scipy.spatial.transform import Rotation as Rot
import time
from numpy.linalg import norm
import math

# Avarage cam 1
cam_cx = (981.4441183 + 12)/1.5
cam_cy = (529.055484 - 11.2)/1.5
cam_fx = (1392.240922)/1.5
cam_fy = (1392.240922)/1.5 * 0.99821077
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

num_objects = 2
objlist = [1, 2]
num_points = 1000
iteration = 10
bs = 1
    
xmap = np.array([[j for i in range(1280)] for j in range(720)])
ymap = np.array([[i for i in range(1280)] for j in range(720)])
cam_scale = 1.0
    
# idx from 0 to 6
#idx = 0
    
# object model points
model_points_list = []
for idx in range(0, 2): 
    model_points = posenet_deploy.get_model_points(idx)
    model_points = model_points[0].cpu().detach().numpy()
    model_points_list.append(model_points)

model = "/home/tomo/densefusion/trained_models/tomo_blisters_2/pose_model_1_0.005430417195209802.pth"
refine_model = "/home/tomo/densefusion/trained_models/tomo_blisters_2/pose_refine_model_63_0.0005920357997828468.pth"


#model = "/home/tomo/densefusion/trained_models/tomo_blisters_invert/pose_model_1_0.005256925600131684.pth"
#refine_model = "/home/tomo/densefusion/trained_models/tomo_blisters_invert/pose_refine_model_29_0.0005998273272780352.pth"


estimator = PoseNet(num_points = num_points, num_obj = num_objects)
estimator.cuda()
refiner = PoseRefineNet(num_points = num_points, num_obj = num_objects)
refiner.cuda()
estimator.load_state_dict(torch.load(model))
refiner.load_state_dict(torch.load(refine_model))
estimator.eval()
refiner.eval()

def setup_config():
    model_path = "/home/tomo/densefusion/MASK_RCNN_TOMO/trained_models/model_final_new.pth"
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    cfg.INPUT.MIN_SIZE_TEST = 0
    cfg.INPUT.MAX_SIZE_TEST = 0
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 2000
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8, 16, 32, 64, 128]]
    cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7 #9
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    return cfg

def setup_realsense():
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    return config

def bbox_convert(bboxes):
    from detectron2.structures import BoxMode
    
    bboxes_converted = BoxMode(0).convert(bboxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS) # XYXY -> XYWH
    return bboxes_converted
    

def iou_score(boxA, boxB): # bounding box format: XYXY
    # determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou


def check_inverted(cropped_image):
    hsv_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2HSV)
    s_image = hsv_image[:,:,1]    
    #s_image = cv2.medianBlur(s_image, 5)

    # _, binary = cv2.threshold(s_image, 127, 255, cv2.THRESH_BINARY)
    binary = cv2.inRange(s_image, 100, 255)
    _, counts = np.unique(binary, return_counts=True)
    
    print(counts)
    if len(counts) == 2  and counts[1] > 200:
        return False    # not inverted blisters
    else:
        return True

    
def main():
    cfg = setup_config()
    pipeline = rs.pipeline()
    realsense_cfg = setup_realsense()
    pipeline.start(realsense_cfg)  # Start streaming
    visualizer =  predictor.VisualizationDemo(cfg)

    ref_frame_axies = []
    ref_frame_label = []
    min_distance = 0.9
    label_cnt = 0
    frameth = 0 

    my_t_pool = {}
    my_r_pool = {}

    while True:
        frameth += 1
        cur_frame_axies = []
        cur_frame_label = []
        my_t_per_frame = []
        my_r_per_frame = []

        align = rs.align(rs.stream.color)
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        rgb = aligned_frames.get_color_frame()
        rgb = np.asanyarray(rgb.get_data())
        frame = rgb.copy()
            
            
        # Do instance segmentation
        start = time.time()
        segmentation, vis = visualizer.run_on_image(frame)
        #print("Time = " + str(time.time()-start))
        
        cv2.imshow('Mask', vis)
        cv2.waitKey(1)
            
        # Get segmentation mask
        ori_label = segmentation['instances'].pred_masks.cpu().numpy()
        label = np.sum(ori_label, axis=0).astype(np.uint8)
        label = np.where(label != 0, 255, label)
        label = Image.fromarray(label).convert("L")
        label = np.asarray(label.convert('RGB')).astype(np.uint8)
        
        bboxes = segmentation['instances'].pred_boxes.tensor.cpu().numpy()
        xyxy_bboxes = bboxes
        bboxes = bbox_convert(bboxes)

        if len(bboxes) > 0:
            #depth_frames = frames.get_depth_frame()
            depth_frames = aligned_frames.get_depth_frame()
                
            video_profile = depth_frames.profile.as_video_stream_profile()
            intr = video_profile.get_intrinsics()
            depth = np.asanyarray(depth_frames.get_data())
            #centers = segmentation['instances'].pred_boxes.get_centers()
            if len(my_t_pool) > 0:
                last_key = list(my_t_pool.keys())[-1]
    
            for i in range(0,len(bboxes)):
                bbox_xyxy = np.array(list(xyxy_bboxes[i]))
                bbox = list(bboxes[i])
                print("Bounding Box:" + str(bbox))
                #center = bboxes[i].get_centers()
                #center = centers[i].cpu().numpy()
                num_idx = float('nan')
                max_value = 0


                label_of_object = ori_label[i].astype(np.uint8)
                label_of_object = np.where(label_of_object != 0, 255, label_of_object)
                label_of_object = Image.fromarray(label_of_object).convert("L")
                label_of_object = np.asarray(label_of_object.convert('RGB')).astype(np.uint8)

                if len(ref_frame_label) > 0:
                    iou_list = []
                    b = bbox_xyxy
                    a = np.array(ref_frame_axies)
                    for k in range(len(ref_frame_axies)):
                        iou = iou_score(a[k], b)
                        iou_list.append(iou)
                    iou_list = np.array(iou_list)
                    max_value = iou_list.max()
                    if(max_value > min_distance):
                        min_idx = np.where(iou_list==max_value)[0][0]
                        num_idx = ref_frame_label[min_idx]
                            
                if(math.isnan(num_idx)):
                    num_idx = label_cnt
                    label_cnt += 1
                cur_frame_label.append(num_idx)
                cur_frame_axies.append(bbox_xyxy)
                
                print(max_value)
                if (frameth == 1) or (max_value < 0.9) or (i > len(my_t_pool[last_key]) - 1) or (frameth % 20 == 0):
                    pos_text = (bbox[0], bbox[1])

                    class_id = segmentation['instances'].pred_classes[i].cpu().data.numpy()
                    print("Class: " + str(class_id))
                    #idx = class_id
                    if class_id == 0:
                        idx = 0
                    if class_id == 2:
                        idx = 1

                    model_points = model_points_list[idx]

                    mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
                    #mask_label = ma.getmaskarray(ma.masked_equal(label, np.array(255)))
                    mask_label = ma.getmaskarray(ma.masked_equal(label_of_object, np.array([255, 255, 255])))[:, :, 0]
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
                    # print(cloud.shape)

                    # cropped img
                    #img_masked = rgb[:, :, :3]
                    img_masked = rgb[:, :, ::-1] # bgr to rgb
                    img_masked = np.transpose(img_masked, (2, 0, 1))
                    img_masked = img_masked[:, rmin:rmax, cmin:cmax]

                    my_mask = np.transpose(label_of_object, (2, 0, 1))
                    my_mask = my_mask[:, rmin:rmax, cmin:cmax]  ## Added by me to crop the mask
                    mask_img = np.transpose(my_mask, (1, 2, 0))
                    img_rgb = np.transpose(img_masked, (1, 2, 0))
                    croped_img_mask = cv2.bitwise_and(img_rgb, mask_img)
                    crop_image_to_check = croped_img_mask.copy()
                    cv2.imshow("mask_crop", croped_img_mask)
                    croped_img_mask = np.transpose(croped_img_mask, (2, 0, 1))

            
                    # Variables
                    cloud = torch.from_numpy(cloud.astype(np.float32)).unsqueeze(0)
                    choose = torch.LongTensor(choose.astype(np.int32)).unsqueeze(0)
                    #img_masked = torch.from_numpy(img_masked.astype(np.float32)).unsqueeze(0)
                    img_masked = torch.from_numpy(croped_img_mask.astype(np.float32)).unsqueeze(0)
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
                
                        my_r_matrix = quaternion_matrix(my_r)[:3, :3]
                    #print("Time = " + str(time.time()-start))
                    my_t_per_frame.append(my_t)
                    my_r_per_frame.append(my_r_matrix)

                    #rotation = Rot.from_matrix(my_r_matrix)
                    #angle = rotation.as_euler('xyz', degrees=True)
		
                    my_t = np.around(my_t, 5)
                    #print("translation vector = " + str(my_t))
                    #print("rotation angles = " + str(my_r))
                
                    frame = posenet_deploy.get_3d_bbox(frame, model_points, my_r_matrix, my_t)
                    frame = posenet_deploy.draw_axes(frame, my_r_matrix, my_t)

                    if check_inverted(crop_image_to_check):
                        cv2.putText(frame, str(num_idx) + "_inverted", pos_text, cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0,255,0), 2, cv2.LINE_AA)
                    else:
                        cv2.putText(frame, str(num_idx), pos_text, cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0,255,0), 2, cv2.LINE_AA)
                    
                    #cv2.putText(frame, str(num_idx), pos_text, cv2.FONT_HERSHEY_SIMPLEX,
                    #            0.5, (0,255,0), 2, cv2.LINE_AA)

                    posenet_deploy.putText(frame, i, num_idx, class_id, my_t)
                    #cv2.imshow('Result', rgb)
                    #cv2.waitKey(1)

                else:
                    rmin, rmax, cmin, cmax = posenet_deploy.get_bbox(bbox)
                    img_masked = rgb[:, :, ::-1] # bgr to rgb
                    img_masked = np.transpose(img_masked, (2, 0, 1))
                    img_masked = img_masked[:, rmin:rmax, cmin:cmax]

                    my_mask = np.transpose(label_of_object, (2, 0, 1))
                    my_mask = my_mask[:, rmin:rmax, cmin:cmax]  ## Added by me to crop the mask
                    mask_img = np.transpose(my_mask, (1, 2, 0))
                    img_rgb = np.transpose(img_masked, (1, 2, 0))
                    croped_img_mask = cv2.bitwise_and(img_rgb, mask_img)
                    crop_image_to_check = croped_img_mask.copy()

                    pos_text = (bbox[0], bbox[1])
                    last_key = list(my_t_pool.keys())[-1]

                    print("POOL: " + str(my_t_pool[last_key]))
                    class_id = segmentation['instances'].pred_classes[i].cpu().data.numpy()
                    
                    my_t = my_t_pool[last_key][min_idx]
                    my_r_matrix = my_r_pool[last_key][min_idx]
                    
                    my_t_per_frame.append(my_t)
                    my_r_per_frame.append(my_r_matrix)
					
                    frame = posenet_deploy.get_3d_bbox(frame, model_points, my_r_matrix, my_t)
                    frame = posenet_deploy.draw_axes(frame, my_r_matrix, my_t)

                    if check_inverted(crop_image_to_check):
                        cv2.putText(frame, str(num_idx) + "_inverted", pos_text, cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0,255,0), 2, cv2.LINE_AA)
                    else:
                        cv2.putText(frame, str(num_idx), pos_text, cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0,255,0), 2, cv2.LINE_AA)
                    
                    #cv2.putText(frame, str(num_idx), pos_text, cv2.FONT_HERSHEY_SIMPLEX,
                    #            0.5, (0,255,0), 2, cv2.LINE_AA)

                    posenet_deploy.putText(frame, i, num_idx, class_id, my_t)

            if len(my_t_per_frame) > 0:
                my_t_pool[frameth] = my_t_per_frame
                my_r_pool[frameth] = my_r_per_frame      
  
            ref_frame_label = cur_frame_label
            ref_frame_axies = cur_frame_axies
            
            end = time.time() - start
            cv2.putText(frame, "Time processing: " + str(round(end, 3)) + " seconds", (100,700), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0,0,255), 2, cv2.LINE_AA)
            cv2.imshow('Result', frame)
            cv2.waitKey(1)
        
        else:
            # Show images
            #video_writer.write(rgb)
            cv2.imshow('Result', rgb)
            cv2.waitKey(1)

    pipeline.stop()


if __name__ == "__main__":
    main()
