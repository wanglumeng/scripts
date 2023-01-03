import numpy as np
import os
import math
from scipy.spatial.transform import Rotation as R
import json
import cv2
import argparse
def getFileList(dir,Filelist, ext=None):

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

def roty(t):
    """ Rotation about the y-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def project_to_image(pts_3d, P):
    """ Project 3d points to image plane.

    Usage: pts_2d = projectToImage(pts_3d, P)
      input: pts_3d: nx3 matrix
             P:      3x4 projection matrix
      output: pts_2d: nx2 matrix

      P(3x4) dot pts_3d_extended(4xn) = projected_pts_2d(3xn)
      => normalize projected_pts_2d(2xn)

      <=> pts_3d_extended(nx4) dot P'(4x3) = projected_pts_2d(nx3)
          => normalize projected_pts_2d(nx2)
    """
    n = pts_3d.shape[0]
    pts_3d_extend = np.hstack((pts_3d, np.ones((n, 1))))
    # print(('pts_3d_extend shape: ', pts_3d_extend.shape))
    pts_2d = np.dot(pts_3d_extend, np.transpose(P))  # nx3
    #print("pts_3d_extend:", pts_3d_extend)
    #print("P :", np.transpose(P))
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    return pts_2d[:, 0:2]

def compute_box_3d(obj, P):
    """ Takes an object and a projection matrix (P) and projects the 3d
        bounding box into the image plane.
        Returns:
            corners_2d: (8,2) array in left image coord.
            corners_3d: (8,3) array in in rect camera coord.
    """
    # compute rotational matrix around yaw axis
    R = roty(obj[14])

    # 3d bounding box dimensions
    l = obj[10]
    w = obj[9]
    h = obj[8]

    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    # print corners_3d.shape
    corners_3d[0, :] = corners_3d[0, :] + obj[11]
    corners_3d[1, :] = corners_3d[1, :] + obj[12]
    corners_3d[2, :] = corners_3d[2, :] + obj[13]
    # print 'cornsers_3d: ', corners_3d
    # only draw 3d bounding box for objs in front of the camera
    if np.any(corners_3d[2, :] < 0.1):
        corners_2d = None
        return corners_2d, np.transpose(corners_3d)

    # project the 3d bounding box into the image plane
    corners_2d = project_to_image(np.transpose(corners_3d), P)
    # print 'corners_2d: ', corners_2d
    return corners_2d, np.transpose(corners_3d)

def draw_projected_box3d(image, qs, color=(0, 255, 0), thickness=2):
    """ Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    """
    qs = qs.astype(np.int32)
    for k in range(0, 4):
        # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
        i, j = k, (k + 1) % 4
        # use LINE_AA for opencv3
        # cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.CV_AA)
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
        i, j = k + 4, (k + 1) % 4 + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)

        i, j = k, k + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)

    # fill
    img1 = np.copy(image) 
    # pts = np.array([ [ [qs[2, 0], qs[2, 1]],[qs[3, 0], qs[3, 1]],[qs[7, 0], qs[7, 1]],[qs[6, 0], qs[6, 1]] ] ], dtype=np.int32) # back
    pts = np.array([ [ [qs[0, 0], qs[0, 1]],[qs[1, 0], qs[1, 1]],[qs[5, 0], qs[5, 1]],[qs[4, 0], qs[4, 1]] ] ], dtype=np.int32) # front    
    cv2.fillPoly(img1, pts, color)
    image = cv2.addWeighted(img1, 0.3, image, 0.7, 0) # transparency

    return image

def draw_projected_box2d(image, box, color=(0, 255, 255), thickness=2):
    # box = box.astype(np.int32)
    cv2.rectangle(image, (int(box[0]),int(box[1])), (int(box[2]),int(box[3])),color, thickness)

def compute_box_bev(box):
    #box:x,y,w,h,roty
    x,y,w,h,roty = box[0], box[1], box[2], box[3], box[4]
    corners = np.array([[w/2, -h/2],[w/2, h/2], [-w/2, h/2],[-w/2, -h/2]])
    c = np.cos(roty)
    s = np.sin(roty)
    R = np.array([[c, -s],[s, c]])
    corners_new = R @ corners.T 
    corners_new = corners_new.T + np.array([x,y])
    return corners_new

class BevCoords:
    TOP_Y_MIN = -60
    TOP_Y_MAX = +60
    TOP_X_MIN = -150
    TOP_X_MAX = 150
    TOP_X_DIVISION = 0.4
    TOP_Y_DIVISION = 0.4
    @staticmethod
    def veh_to_bev_coords(x, y):
        # print("TOP_X_MAX-TOP_X_MIN:",TOP_X_MAX,TOP_X_MIN)
        Xn = int((BevCoords.TOP_X_MAX - BevCoords.TOP_X_MIN) // BevCoords.TOP_X_DIVISION) + 1
        Yn = int((BevCoords.TOP_Y_MAX - BevCoords.TOP_Y_MIN) // BevCoords.TOP_Y_DIVISION) + 1
        # xx = Yn - int((y - TOP_Y_MIN) // TOP_Y_DIVISION)
        # yy = Xn - int((x - TOP_X_MIN) // TOP_X_DIVISION)
        xx = int((y - BevCoords.TOP_Y_MIN) // BevCoords.TOP_Y_DIVISION)
        yy = Xn - int((x - BevCoords.TOP_X_MIN) // BevCoords.TOP_X_DIVISION)

        return 300 - xx, yy


def show_topview_with_boxes(top_image, boxes, color=(255, 0, 0), thickness=1):
    for box in boxes:
        corner = compute_box_bev(box).tolist()
        corner = [BevCoords.veh_to_bev_coords(xy[0], xy[1]) for xy in corner]
        for i in range(4):
            cv2.line(top_image, (corner[i][0], corner[i][1]), ((corner[(i+1)%4][0], corner[(i+1)%4][1])), color, thickness, cv2.LINE_AA)
    cv2.circle(top_image, (int(top_image.shape[1] /2), int(top_image.shape[0] /2)), 5, (0, 255, 0), 0)
    cv2.line(top_image, (int(top_image.shape[1] /2) +5, 0), (int(top_image.shape[1] /2) +5, 750), (0,255,255), thickness, cv2.LINE_AA)
    cv2.line(top_image, (int(top_image.shape[1] /2) -5, 0), (int(top_image.shape[1] /2) -5, 750), (0,255,255), thickness, cv2.LINE_AA)

class labelTranslator:
    def __init__(self, calib_file):
        with open(calib_file) as fi:
            lines = fi.readlines()           
            # assert (len(lines) == 12)
        rd = math.pi/180
        for  line in lines:
            line = line.strip().split(' ')
            line[1:] = [float(i) for i in line[1:]]
            line[0] = line[0][:-1]
            if "inter" in line[0]:
                self.__dict__[line[0]] = np.array([[line[1], 0, line[2]], [0, line[3], line[4]], [0, 0, 1]])
                self.__dict__[line[0] + "_distortion"] = np.array(line[5:])
            if "extri" in line[0]:
                roll = line[3]*rd
                roll_mat = np.array([[1,0,0],[0,math.cos(roll), -math.sin(roll)], [0,math.sin(roll), math.cos(roll)]])
                pitch = line[2]*rd
                pitch_mat = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0,1,0],[-math.sin(pitch), 0, math.cos(pitch)]])
                yaw = line[1]*rd
                yaw_mat = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0],[0,0,1]])
                self.__dict__[line[0] + "_rotation"] = yaw_mat @ pitch_mat @ roll_mat
                self.__dict__[line[0] + "_translation"] = np.array(line[4:])
    
    def GetLabel(self, category):
        if category == "truck":
            return 'Truck'
        elif category == "car":
            return 'Car'
        elif category == "van":
            return 'Van'
        elif category == "bus":
            return 'Van'
        else :
            return 'Ignore'
    
    def roty2cam(self, rotz, sensor_name):
        cam2veh_rotation = getattr(self, "camera_{}_extri_rotation".format(sensor_name))
        ori_y = np.linalg.inv(cam2veh_rotation) @ np.array([[math.cos(rotz)],[math.sin(rotz)],[0]])
        roty = -math.atan2(ori_y[2], ori_y[0])
        #roty = -(rotz - self.extri_param['yaw']*math.pi/180) #+ math.pi/2
        if roty > math.pi:
            roty -= 2*math.pi
        if roty < -math.pi:
            roty += 2*math.pi
        return roty
    
    def cvtobj(self, obj_in, sensor_name):
        x, y, z = obj_in[1], obj_in[2], obj_in[3]
        label = self.GetLabel(obj_in[0])
        xyz_veh = np.array([x, y, z])
        cam2veh_rotation = getattr(self, "camera_{}_extri_rotation".format(sensor_name))
        cam2veh_translation = getattr(self, "camera_{}_extri_translation".format(sensor_name))
        xyz_cam = np.matmul(np.linalg.inv(cam2veh_rotation), xyz_veh - cam2veh_translation)
        roty = self.roty2cam(obj_in[7], sensor_name)
        # xyz, alpha, roty = self.CoordinateChange(x, y, z,  obj_in[6], obj_in[7])
        # rect_cen = np.array([( obj_in[8]+ obj_in[10])/2.0,( obj_in[9]+ obj_in[11])/2.0]) 
        # rect_cen_undt = imdt.undistortPts(rect_cen, imdt.F)
        # _,_,beta = calc_ray(P, rect_cen_undt)
        
        # alpha = roty - beta 
        # if alpha > math.pi:
        #     alpha -= 2*math.pi
        # if alpha < -math.pi:
        #     alpha += 2*math.pi
        K = getattr(self, "camera_{}_inter".format(sensor_name))
        xyz_cam = np.linalg.inv(K) @ np.array([(obj_in[8]+obj_in[10])/2, (obj_in[9]+obj_in[11])/2, 1]) * xyz_cam[2]
        xyz_veh = np.matmul(cam2veh_rotation, xyz_cam) + cam2veh_translation
        obj_out = [label, 0, 0, 0, obj_in[8], obj_in[9], obj_in[10], obj_in[11], 
        obj_in[6], obj_in[5], obj_in[4], xyz_veh[0], xyz_veh[1], xyz_veh[2], roty]
        return obj_out

    def cvtobj_kitti(self, obj_in, sensor_name):
        x, y, z = obj_in[1], obj_in[2], obj_in[3]
        label = self.GetLabel(obj_in[0])
        xyz_veh = np.array([x, y, z])
        cam2veh_rotation = getattr(self, "camera_{}_extri_rotation".format(sensor_name))
        cam2veh_translation = getattr(self, "camera_{}_extri_translation".format(sensor_name))
        xyz_cam = np.matmul(np.linalg.inv(cam2veh_rotation), xyz_veh - cam2veh_translation)
        roty = self.roty2cam(obj_in[7], sensor_name)
        K = getattr(self, "camera_{}_inter".format(sensor_name))
        # xyz_cam = np.linalg.inv(K) @ np.array([(obj_in[8]+obj_in[10])/2, (obj_in[9]+obj_in[11])/2, 1]) * xyz_cam[2]
        obj_out = [label, 0, 0, 0, obj_in[8], obj_in[9], obj_in[10], obj_in[11], 
                    obj_in[6], obj_in[5], obj_in[4], xyz_cam[0], xyz_cam[1] + obj_in[6]/2, xyz_cam[2], roty]
        return obj_out

sensor_name_map = {
    "image_0": "l120",
    "image_1": "r120",
    "image_2": "l60",
    "image_3": "r60",
    "image_4": "f120",
    "image_5": "f60",
    "image_6": "f30",
}

def vis_trunk(data_dir, labels_dir, calib_file):
    trans = labelTranslator(calib_file)
    json_files = getFileList(labels_dir, [], ext='json')
    #json_files.sort()
    idx = 1
    # for json_file in json_files:
    while(1):
        json_file = json_files[idx]
        # json_file = "/home/qianlei/rosbag/0313/annations/json_lidar/点云融合新规则交付json7304/20220802-7304/6753f799-c68c-4302-845a-603abc90f67d_1_100/1659423107112643000.json"
        f = open(json_file, encoding="utf-8")
        frames = json.load(f)
        image_list = frames["images"]
        lidar_obj_list =  frames["items"]
        imgs_dict = {}
        img_bev = np.ones((750, 300, 3), np.uint8)*255
        for img in image_list:
            simgname = img['image'].strip().split('annotations/')[-1]
            sensor_name = sensor_name_map[simgname.split("/")[-2]]
            #img_path = os.path.join(data_dir, "20220802_" + simgname.split("zx-data/")[-1])
            img_path = os.path.join(data_dir,  simgname.split("zx-data/")[-1])
            # Distortion correction image
            print(img_path)
            ori_img = cv2.imread(img_path)
            K = getattr(trans, "camera_{}_inter".format(sensor_name))
            distortion= getattr(trans, "camera_{}_inter_distortion".format(sensor_name))
            w,h = ori_img.shape[1], ori_img.shape[0]
            mapx,mapy = cv2.initUndistortRectifyMap(K, distortion, None, K, (w,h), 5)
            ori_img = cv2.remap(ori_img, mapx, mapy, cv2.INTER_LINEAR)
            
            imgitems = img["items"]
            if imgitems == None:
                imgitems = []

            annotations = []
            objs_veh = []
            objs_cam = []
            for img_obj in imgitems:
                for lidar_obj  in lidar_obj_list:          
                    if img_obj["id"] == lidar_obj["id"]:
                        xmin,ymin,xmax,ymax = [float(pp) for pp in img_obj['box2d']]
     

                        # Distortion correction 2D
                        new_point = cv2.undistortPoints(np.array([[xmin,ymin], [xmax,ymax]]), K, distortion, P = K)
                        xmin,ymin, xmax,ymax = new_point[0][0][0], new_point[0][0][1], new_point[1][0][0], new_point[1][0][1]

                        # lidar info 
                        rot_z = lidar_obj['rotation']['z']  #(-pi~pi)
                        label = lidar_obj['category']
                        x, y, z = lidar_obj['position']['x'], lidar_obj['position']['y'], lidar_obj['position']['z'] 
                        cl, cw, ch = lidar_obj['dimension']['x'], lidar_obj['dimension']['y'], lidar_obj['dimension']['z']
                        obj = [label, x, y, z, cl, cw, ch, rot_z, xmin, ymin, xmax, ymax]
                        objs_veh.append([label, 0, 0, 0, xmin, ymin, xmax, ymax, ch, cw, cl, x, y, z, rot_z])
                        obj_out = trans.cvtobj(obj, sensor_name)
                        objs_cam.append(obj_out)
                        annotations.append(trans.cvtobj_kitti(obj, sensor_name))
            
            P2 = np.concatenate((K, [[0],[0],[0]]), axis=1)
            for ann in annotations:
                box3d_pts_2d, _ = compute_box_3d(ann, P2)
                if box3d_pts_2d is not None:
                    draw_projected_box3d(ori_img, box3d_pts_2d, (0,0,255))
                draw_projected_box2d(ori_img, ann[4:8], color=(0, 255, 255), thickness=2)
            imgs_dict[sensor_name] = ori_img
            boxes = [[obj_veh[11],obj_veh[12], obj_veh[10],obj_veh[9], obj_veh[14]] for obj_veh in objs_veh]
            show_topview_with_boxes(img_bev, boxes)
            # w,h = ori_img.shape[1], ori_img.shape[0]
        
        
        bev_w, bev_h = img_bev.shape[1], img_bev.shape[0]
        for key, img in imgs_dict.items():
            ori_w, ori_h = img.shape[1], img.shape[0]
            imgs_dict[key] = cv2.resize(img, (int(ori_w *(bev_h/2 / ori_h)), int(bev_h/2)), interpolation = cv2.INTER_LINEAR) 
        img_joint_l = np.concatenate([imgs_dict["l60"], imgs_dict["l120"]], axis=0) 
        img_joint_r = np.concatenate([imgs_dict["r60"], imgs_dict["r120"]], axis=0) 
        img_joint = np.concatenate([img_joint_l, img_bev, img_joint_r], axis=1)
        img_joint_f = np.concatenate([imgs_dict["f60"], imgs_dict["f120"]], axis= 1 ) 
        img_joint_top = np.ones((img_joint_f.shape[0], img_joint.shape[1], 3), np.uint8)*255
        start = int((img_joint_top.shape[1] -img_joint_f.shape[1])/2)
        img_joint_top[:, start: start+img_joint_f.shape[1]] = img_joint_f
        img_joint =  np.concatenate([img_joint_top, img_joint], axis=0) 
        img_joint = cv2.resize(img_joint, (640, 480), interpolation = cv2.INTER_LINEAR) 
        cv2.imshow("img", img_joint)
        k = cv2.waitKey(0)
        if k == ord('s'): 
            idx +=  1
        if k == ord('a'): 
            idx -=  1
        if k == ord('q'):
            break


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/home/qianlei/rosbag/0313/labeling/", help="kitti data path")
    parser.add_argument("--labels_dir", type=str, default="/home/qianlei/rosbag/0313/annations/json_vision/", help="frame of the data")
    parser.add_argument("--calib_file", type=str, default="/home/qianlei/rosbag/0313/annations/calibration_carmera_0313.txt", help="frame of the data")
    opt = parser.parse_args()

    vis_trunk(opt.data_dir, opt.labels_dir, opt.calib_file)

