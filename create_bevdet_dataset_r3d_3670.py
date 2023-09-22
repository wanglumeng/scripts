# # Copyright (c) OpenMMLab. All rights reserved.
# import pickle

import numpy as np
from nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

from scipy.spatial.transform import Rotation as R
import os
import json

from tqdm import tqdm
from pathlib import Path
from PIL import ExifTags, Image, ImageOps
import pickle
import copy

IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp' 

map_name_from_general_to_detection = {
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.wheelchair': 'ignore',
    'human.pedestrian.stroller': 'ignore',
    'human.pedestrian.personal_mobility': 'ignore',
    'human.pedestrian.police_officer': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'animal': 'ignore',
    'vehicle.car': 'car',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.truck': 'truck',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.emergency.ambulance': 'ignore',
    'vehicle.emergency.police': 'ignore',
    'vehicle.trailer': 'trailer',
    'movable_object.barrier': 'barrier',
    'movable_object.trafficcone': 'traffic_cone',
    'movable_object.pushable_pullable': 'ignore',
    'movable_object.debris': 'ignore',
    'static_object.bicycle_rack': 'ignore',
}
# classes = [
#     'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
#     'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
# ]

classes = [
    'car', 'truck', 'bus'
]


def get_gt(info):
    """Generate gt labels from info.

    Args:
        info(dict): Infos needed to generate gt labels.

    Returns:
        Tensor: GT bboxes.
        Tensor: GT labels.
    """
    ego2global_rotation = info['cams']['CAM_FRONT']['ego2global_rotation']
    ego2global_translation = info['cams']['CAM_FRONT'][
        'ego2global_translation']
    trans = -np.array(ego2global_translation)
    rot = Quaternion(ego2global_rotation).inverse
    gt_boxes = list()
    gt_labels = list()
    for ann_info in info['ann_infos']:
        # Use ego coordinate.
        if (map_name_from_general_to_detection[ann_info['category_name']]
                not in classes
                or ann_info['num_lidar_pts'] + ann_info['num_radar_pts'] <= 0):
            continue
        box = Box(
            ann_info['translation'],
            ann_info['size'],
            Quaternion(ann_info['rotation']),
            velocity=ann_info['velocity'],
        )
        box.translate(trans)
        box.rotate(rot)
        box_xyz = np.array(box.center)
        box_dxdydz = np.array(box.wlh)[[1, 0, 2]]
        box_yaw = np.array([box.orientation.yaw_pitch_roll[0]])
        box_velo = np.array(box.velocity[:2])
        gt_box = np.concatenate([box_xyz, box_dxdydz, box_yaw, box_velo])
        gt_boxes.append(gt_box)
        
        gt_labels.append(
            classes.index(
                map_name_from_general_to_detection[ann_info['category_name']]))
    return gt_boxes, gt_labels


def nuscenes_data_prep(root_path, info_prefix, version, max_sweeps=10):
    """Prepare data related to nuScenes dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        max_sweeps (int, optional): Number of input consecutive frames.
            Default: 10
    """
    nuscenes_converter.create_nuscenes_infos(
        root_path, info_prefix, version=version, max_sweeps=max_sweeps)

def xyxy2xywhn(x, w=640, h=640):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    y = x.copy()
    y[0] = ((x[0] + x[2]) / 2) / w  # x center
    y[1] = ((x[1] + x[3]) / 2) / h  # y center
    y[2] = (x[2] - x[0]) / w  # width
    y[3] = (x[3] - x[1]) / h  # height
    return y

def save_image(orifile, outfile):
    import shutil
    shutil.copy(orifile, outfile)
    
def save_label(yolos, outfile):
    with open(outfile, 'w') as tf:
        tf.writelines(yolos)
        
def save_calib(incoef, extri, outfile):
    intra_coef_str = str(incoef)
    extri_param_str = str(extri)
    calib_lines = [intra_coef_str+'\n', extri_param_str]
    with open(outfile, 'w') as tf:
        tf.writelines(calib_lines)


def extri_param2RT(extri_param):
    rd = np.pi/180
    RT = np.array([[1,0,0,extri_param['x']],[0,1,0,extri_param['y']],[0,0,1,extri_param['z']],[0,0,0,1]])
    roll = extri_param['roll']*rd
    roll_mat = np.array([[1,0,0],[0,np.cos(roll), -np.sin(roll)], [0,np.sin(roll), np.cos(roll)]])
    pitch = extri_param['pitch']*rd
    pitch_mat = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0,1,0],[-np.sin(pitch), 0, np.cos(pitch)]])
    yaw = extri_param['yaw']*rd
    yaw_mat = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0],[0,0,1]])
    rot_mat = yaw_mat @ pitch_mat @ roll_mat
    RT[:-1, :-1] = rot_mat
    Rt_cam2car = RT
    Rt_car2cam = np.linalg.inv(RT)
    q_cam2car = Quaternion(matrix=rot_mat)
    q_car2cam = q_cam2car.inverse
    return Rt_cam2car, Rt_car2cam, q_cam2car, q_car2cam


def compute_3Dbox_cam(self, obj, DISTORT=False):
    def distort_points(corners_2D, DPs):
        k1, k2, p1, p2, k3 = DPs
        r = np.linalg.norm(corners_2D[:2, :],ord=None,axis=0)
        x, y = corners_2D[0, :], corners_2D[1, :]
        corners_2D[0, :] = x*(1+k1*r**2+k2*r**4+k3*r**6) + 2*p1*x*y + p2*(r**2+2*x**2)
        corners_2D[1, :] = y*(1+k1*r**2+k2*r**4+k3*r**6) + 2*p2*x*y + p1*(r**2+2*x**2)
        return corners_2D
    
    label, xmin, ymin, xmax, ymax, rotz, l, w, h, x, y, z = obj
    
    # Draw 3D Bounding Box
    R = np.array([[np.cos(rotz), -np.sin(rotz), 0],
                [np.sin(rotz), np.cos(rotz), 0],
                [0, 0, 1]])
    
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    # y_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    y_corners = [-w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2]
    z_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]

    corners_3D = np.array([x_corners, y_corners, z_corners])
    
    corners_3D = R.dot(corners_3D)
    corners_3D += np.array([x, y, z]).reshape((3, 1))
    
    corners_3D_1 = np.vstack((corners_3D, np.ones((corners_3D.shape[-1]))))
    corners_3D_cam = np.dot(self.Rt_car2cam, corners_3D_1)
    
        
    if DISTORT:
        corners_3D_norm = corners_3D_cam / abs(corners_3D_cam[2])
        corners_3D_cam = distort_points(corners_3D_norm, self.DPs)
        
    # if (corners_3D_cam[2, :] < 0).any():
    #     minc = corners_3D_cam[2, :].min()
    #     corners_3D_cam[2, :][corners_3D_cam[2, :] < 0] -= (minc-4)
    
    corners_2D = np.dot(self.K, corners_3D_cam)
    corners_2D = corners_2D / corners_2D[2]
    corners_2D = corners_2D[:2]
    return corners_2D
    
    
def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]

def img2calib_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'calibs' + os.sep  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]

def img2depth_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'depths' + os.sep  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]

def load_calib(calibf):
    with open(calibf) as f:
        lb = [eval(x) for x in f.read().strip().splitlines() if len(x)]
    Ins = lb[0]
    In_Coef = [[Ins[0], 0., Ins[2]],
            [0., Ins[1], Ins[3]],
            [0., 0., 1.]]
    DPs = Ins[4:]
    intra_coef = lb[0]
    extri_param = lb[1]
    In_Coef = np.array(In_Coef)
    DPs = np.array(DPs)
    Proj = np.linalg.inv(In_Coef)
    Rt_cam2car, Rt_car2cam, q_cam2car, q_car2cam = extri_param2RT(extri_param)
    
    camparams = dict(
        In_Coef=np.float32(In_Coef),
        Proj=np.float32(Proj),
        DPs=np.float32(DPs),
        Rt_cam2car=np.float32(Rt_cam2car),
        Rt_car2cam=np.float32(Rt_car2cam),
        intra_coef=intra_coef,
        extri_param=extri_param,
        q_cam2car=q_cam2car,
        q_car2cam=q_car2cam
    )
    return camparams

def load_images(timestamp, images, bagname):
    camd = {}
    for image in images:
        fname = os.path.basename(image['image'])
        cam = image['image'].split('/')[-2].split('_')[-1]
        fname = fname.replace('time', cam).split('.')[0]
        depth_name = timestamp
        
        labels_dir = f'{dataset}/{bagname}/{cam}/labels/{fname}.txt'
        images_dir = f'{dataset}/{bagname}/{cam}/images/{fname}.jpg'
        calibs_dir = f'{dataset}/{bagname}/{cam}/calibs/{fname}.txt'
        depths_dir = f'{dataset}/{bagname}/{cam}/depths/{cam}_{depth_name}.bin'
        if os.path.isfile(images_dir) and os.path.isfile(calibs_dir):
            
            camparams = load_calib(calibs_dir)
            
            camd[cam] = dict(
                data_path=f'{out_path}/{bagname}/{cam}/images/{fname}.jpg',
                type=cam,
                sample_data_token=timestamp,
                timestamp=int(timestamp),
                cam_intrinsic=camparams["In_Coef"],
                sensor2lidar_rotation=camparams["q_cam2car"].q.tolist(),
                sensor2lidar_translation=camparams["Rt_cam2car"][:3, 3:].squeeze(-1).tolist(),
                sensor2ego_rotation=camparams["q_cam2car"].q.tolist(),
                sensor2ego_translation=camparams["Rt_cam2car"][:3, 3:].squeeze(-1).tolist(),
                ego2global_rotation=[],
                ego2global_translation=[],
            )
        else:
            return 
        
    return camd

def load_gt(items):
    lidar_boxes = []
    lidar_names = []
    num_lidar_pts = []
    valid_flag = []
    gt_boxes = []
    gt_labels = []
    for item in items:
        category = item["category"]
        if category not in classes:
            if category in map_class:
                category = map_class[category]
            else:
                continue
        position = item["position"]
        dimension = item["dimension"]
        rotation = item["rotation"]
        # quaternion = item["quaternion"]
        pointCount = item["pointCount"]["lidar"]
        lidar_box = [position['x'], position['y'], position['z'], dimension['x'], dimension['y'], dimension['z'], rotation['z']]
        lidar_boxes.append(lidar_box)
        lidar_names.append(category)
        num_lidar_pts.append(pointCount)
        valid = False if category == 'ignore' else True
        valid_flag.append(valid)
        if valid:
            gt_box = [position['x'], position['y'], position['z'], dimension['x'], dimension['y'], dimension['z'], rotation['z'], 0., 0.]
            gt_label = classes.index(category)
            gt_boxes.append(gt_box)
            gt_labels.append(gt_label)
    return np.array(lidar_boxes), np.array(lidar_names), np.array(num_lidar_pts), np.array(valid_flag), np.array(gt_boxes), np.array(gt_labels)


def create_dir_not_exist(path, RM=False):
    import shutil
    if os.path.isdir(path):
        if RM:
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)

def getFileList(dir,Filelist, ext=None):
    """
    获取文件夹及其子文件夹中文件列表
    输入 dir：文件夹根目录
    输入 ext: 扩展名
    返回： 文件路径列表
    """
    newDir = dir
    if os.path.isfile(dir):
        if ext is None:
            Filelist.append(dir)
        else:
            if ext in dir[-4:]:
                Filelist.append(dir)
    
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir=os.path.join(dir,s)
            getFileList(newDir, Filelist, ext)
    return Filelist

def process_one(jsonf):
    jsonl = jsonf.split('/')
    bagnl = jsonl[jsonl.index('jsons') + 1]
    bagname = bagnl.split('*')[-1]
    with open(jsonf, 'r') as jf:
        cont = json.load(jf)
    frameUrl = cont['frameUrl']
    images = cont['images']
    items = cont['items']
    timestamp = os.path.basename(frameUrl).split('_')[-1].split('.')[0]
    lidar_boxes, lidar_names, num_lidar_pts, valid_flag, gt_boxes, gt_labels = load_gt(items)
    valid_flag = valid_flag.astype(bool)
    camd = load_images(timestamp, images, bagname)
    
    ann_infos = [gt_boxes, gt_labels]
    
    info = dict(
        lidar_path=f'{out_path}/{bagname}/pcd_all/time_{timestamp}.pcd',
        token=timestamp,
        cams=camd,
        timestamp=int(timestamp),
        lidar2ego_translation=[],
        lidar2ego_rotation=[],
        ego2global_translation=[],
        ego2global_rotation=[],
        gt_boxes=lidar_boxes,
        gt_names=lidar_names,
        num_lidar_pts=num_lidar_pts,
        valid_flag=valid_flag,
        ann_infos=ann_infos
    )
    return info
    

if __name__ == '__main__':
    json_path = '/home/fenglongfei/JSdisk/data/r3d_ori/jsons'
    
    dataset = '/home/fenglongfei/NewDisk/data/r3d/3670/ori/DATASETS/'
    out_pickle = "/home/fenglongfei/NewDisk/data/r3d/3670/ori/DATASETS/bevdet-r3d_infos_train_ori_whole_3cls.pkl"
    out_path = '/ssd/data/r3d/3670/ori/DATASETS/'
    
    # nuscenes_data_prep(
    #     root_path=root_path,
    #     info_prefix=extra_tag,
    #     version=train_version,
    #     max_sweeps=0)
    
    
    # classes = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
    map_class = {'tricycle': 'car'}                                                                                             
    # catd = {'car': 0, 
    #         'truck': 1, 
    #         'bus': 2, 
    #         'trailer': 1, 
    #         'construction_vehicle': 1}
    
    caml = ['f60', 'f120', 'l120', 'r120', 'l60', 'r60']
    
    RM = False
        
    jsons = getFileList(json_path, [], 'json')
    
    infos = []
    
    for jsonf in tqdm(jsons):
        info = process_one(jsonf)
        infos.append(info)
        # if len(infos) > 100:
        #     break

    pklinfo = dict(
        infos=infos,
        metadata = {'version': 'v1.0-trainval'},
    )
    with open(out_pickle, 'wb') as fid:
        pickle.dump(pklinfo, fid)
    
    
    
    print()

# import pickle

# dataset = pickle.load(
#             open('/home/fenglongfei/NewDisk/data/r3d/3670/und/DATASETS/out18/bevdetv2-nuscenes_infos_train_1000.pkl', 'rb'))
# print()