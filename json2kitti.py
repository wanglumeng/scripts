import json
import os
import numpy as np
import math
from im_distortion import ImDistortion


class labelTranslator:
    def __init__(self, K, extri_param):
        self.extri_param2RT(extri_param)
        self.K = K
        self.K = np.concatenate((self.K, [[0], [0], [0]]), axis=1)
        pitch = math.asin(self.Rt[2, 2])
        self.Pitch = np.array([[1, 0, 0, 0], [0, math.cos(
            pitch), -math.sin(pitch), 0], [0, math.sin(pitch), math.cos(pitch), 0], [0, 0, 0, 1]])
        #self.Pitch = np.array([[1,0,0,0],[0,1,0,0],[0,0.0,1,0],[0,0,0,1]])
        self.inv_Pitch = self.Pitch.transpose()

    def extri_param2RT(self, extri_param):
        rd = math.pi/180
        RT = np.array([[1, 0, 0, extri_param['x']], [0, 1, 0, extri_param['y']], [
                      0, 0, 1, extri_param['z']], [0, 0, 0, 1]])
        roll = extri_param['roll']*rd
        roll_mat = np.array([[1, 0, 0], [0, math.cos(
            roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]])
        pitch = extri_param['pitch']*rd
        pitch_mat = np.array([[math.cos(pitch), 0, math.sin(pitch)], [
                             0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]])
        yaw = extri_param['yaw']*rd
        yaw_mat = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                           [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
        rot_mat = yaw_mat @ pitch_mat @ roll_mat
        RT[:-1, :-1] = rot_mat
        # RT = np.array([[0.421906, 0.0860292, -0.902549, 5.32],[0.905606, 0.00752286, 0.424053, 1.4],[0.0432707, -0.996264, -0.0747346, 2.16],[0,0,0,1]])#l60
        self.Rt = np.linalg.inv(RT)
        self.yaw_mat = np.linalg.inv(yaw_mat)
        self.pitch_mat = np.linalg.inv(pitch_mat)
        self.roll_mat = np.linalg.inv(roll_mat)

    def CoordinateChange(self, x, y, z, obj_h, rotz=float):
        ori_y = self.Rt @ np.array([[math.cos(rotz)],
                                   [math.sin(rotz)], [0], [0]])
        roty = -math.atan2(ori_y[2], ori_y[0])
        # roty = -(rotz - self.extri_param['yaw']*math.pi/180) #+ math.pi/2
        if roty > math.pi:
            roty -= 2*math.pi
        if roty < -math.pi:
            roty += 2*math.pi
        #'lidar coordinate to camera coordinate'
        xyz_lidar = np.array([[x], [y], [z], [1]])
        xyz = np.dot(self.Pitch, np.dot(self.Rt, xyz_lidar))
        alpha = -math.atan(xyz[0]/xyz[2]) + roty
        #alpha = -alpha
        if alpha > math.pi:
            alpha -= 2*math.pi
        if alpha < -math.pi:
            alpha += 2*math.pi
        xyz_lidar[2, 0] -= obj_h/2
        xyz = np.dot(self.Pitch, np.dot(self.Rt, xyz_lidar))
        return xyz.tolist(), alpha, roty

    def GetLabel(self, category):
        if category == "truck":
            return 'Truck'
        elif category == "car":
            return 'Car'
        elif category == "van":
            return 'Van'
        elif category == "bus":
            return 'Van'
        else:
            return 'Ignore'

    def cvtobj(self, obj_in, P, imdt):
        x = obj_in[1]
        y = obj_in[2]
        z = obj_in[3]
        label = self.GetLabel(obj_in[0])
        xyz, alpha, roty = self.CoordinateChange(
            x, y, z,  obj_in[6], obj_in[7])
        rect_cen = np.array([(obj_in[8] + obj_in[10])/2.0,
                            (obj_in[9] + obj_in[11])/2.0])
        rect_cen_undt = imdt.undistortPts(rect_cen, imdt.F)
        _, _, beta = calc_ray(P, rect_cen_undt)
        alpha = roty - beta  # 用目标图像中心计算 alpha
        if alpha > math.pi:
            alpha -= 2*math.pi
        if alpha < -math.pi:
            alpha += 2*math.pi
        obj_out = [label, 0, 0, alpha, obj_in[8], obj_in[9], obj_in[10], obj_in[11],
                   obj_in[6], obj_in[5], obj_in[4], xyz[0][0], xyz[1][0], xyz[2][0], roty]
        return obj_out


def create_dir_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def calc_ray(P, pt):
    pt = np.append(pt, [1])
    ray = np.dot(np.linalg.inv(P[:, :3]), pt)
    ray = ray/np.linalg.norm(ray)
    t_org = -np.dot(np.linalg.inv(P[:, :3]), P[:, 3])
    beta = math.atan(ray[0]/ray[2])
    return ray, t_org, beta  # center point 3D direction, origin, beta


def xyxy2xywh(x, w=1280, h=720):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    y0 = ((x[0] + x[2]) / 2) / w  # x center
    y1 = ((x[1] + x[3]) / 2) / h  # y center
    y2 = (x[2] - x[0]) / w  # width
    y3 = (x[3] - x[1]) / h  # height
    return y0, y1, y2, y3


def cvtjson2txt(jsonfile, trans, out_dir, intra_coef):
    P2 = np.dot(trans.K, trans.inv_Pitch)
    imdt = ImDistortion(intra_coef, 1280, 720)

    create_dir_not_exist(out_dir+'label_2/')
    create_dir_not_exist(out_dir+'calib/')
    create_dir_not_exist(out_dir+'image_2/')
    create_dir_not_exist(out_dir+'label_yolo/')

    f = open(jsonfile, encoding="utf-8")
    frames = json.load(f)
    frameId = frames["frameId"]
    # print(frameId)
    frameUrl = frames["frameUrl"]
    pcdfilename = frameUrl.split('/')[-1].split('.')[0].split('_')[-1]
    # print(pcdfilename)
    lidar_obj_list = frames["items"]

    image_list = frames["images"]

    objs = []
    yolos = []
    for img in image_list:
        if not 'f60' in img['image']:  # 只处理l60相机
            continue
        sfoldimg = img['image'].split('/')[-2]
        if not os.path.exists(sfoldimg):
            os.makedirs(sfoldimg)

        simgname = img['image'].split('/')[-1].split(".")[0]
        imgitems = img["items"]
        for img_obj in imgitems:
            for lidar_obj in lidar_obj_list:
                if img_obj["id"] == lidar_obj["id"]:
                    xmin = float(img_obj['position']['x'])
                    ymin = float(img_obj['position']['y'])
                    xmax = xmin + float(img_obj['dimension']['x'])
                    ymax = ymin + float(img_obj['dimension']['y'])

                    # lidar info
                    rot_z = lidar_obj['rotation']['z']  # (-pi~pi)
                    # rot_y = -rot_z
                    # print(lidar_obj['category'])
                    label = lidar_obj['category']
                    x = lidar_obj['position']['x']
                    y = lidar_obj['position']['y']
                    z = lidar_obj['position']['z']

                    cl = lidar_obj['dimension']['x']
                    cw = lidar_obj['dimension']['y']
                    ch = lidar_obj['dimension']['z']
                    obj = [label, x, y, z, cl, cw, ch,
                           rot_z, xmin, ymin, xmax, ymax]

                    obj_out = trans.cvtobj(obj, P2, imdt)
                    objs.append(obj_out)
                    if label.lower() in cls_dic:
                        cls = cls_dic[label.lower()]
                        x_pixel, y_pixel, w_pixel, h_pixel = xyxy2xywh(
                            [xmin, ymin, xmax, ymax])
                        yolo = [cls, x_pixel, y_pixel, w_pixel,
                                h_pixel, obj_out[3], cl, cw, ch, x, y, z]
                        yolo = map(str, yolo)
                        yolos.append(' '.join(yolo) + '\n')

        file_yolo = open(out_dir+'label_yolo/{}.txt'.format(simgname), 'w')
        file_yolo.writelines(yolos)
        file_yolo.close()

        # write label txt

        # file_lb=open(out_dir+'label_2/{}.txt'.format(simgname),'w')
        # for obj in objs:
        #    for item in obj:
        #        file_lb.write(str(item)+' ')
        #    file_lb.write('\n')
        # file_lb.close()
        # #write calibration txt
        # P2 = P2.reshape(1, 12).tolist()
        # file_cal=open(out_dir+'calib/{}.txt'.format(simgname),'w')
        # file_cal.write('P2:')
        # for item in P2[0]:
        #     file_cal.write(' '+ str(item))
        # file_cal.write('\n')
        # file_cal.write('In_Coef:')
        # for item in intra_coef:
        #     file_cal.write(' '+ str(item))
        # file_cal.close()
        # #image dir
        # os.system('cp ' + img_in_dir +'{}.jpg'.format(simgname) + ' ' + out_dir + 'image_2/' + '{}.jpg'.format(simgname))


def getFileList(dir, Filelist, ext=None):
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
            newDir = os.path.join(dir, s)
            getFileList(newDir, Filelist, ext)

    return Filelist


if __name__ == '__main__':
    out_dir = '/media/trunk/sata/datasets/TRUNK/0802/kitti/f60/'
    if os.path.exists(out_dir):
        os.system(f'rm -r {out_dir}')
        print(f'rm -r {out_dir}')
    # img_in_dir = '/home/trunk/workspace/3d/jss/fs/camera_f120'
    extri_param = {'yaw': 89.17, 'pitch': 178.40,
                   'roll': 87.43, 'x': 4.17, 'y': 0.18, 'z': 2.43}
    K = np.array([[1361.619212393047, 0, 660.8204792553835], [
                 0, 1362.499788592409, 244.3957104029807], [0, 0, 1]])
    intra_coef = [1361.619212393047, 1362.499788592409, 660.8204792553835, 244.3957104029807, -
                  0.5993876749576975, 0.4204803723239471, 0.01705854301089264, 0.003017995559993518, -0.3618606861029759]

    cls_dic = {
        "car": 0,
        "truck": 1,
        "van": 2,
        "bus": 2
    }

    # json_dir = '../1月5日交付_3420-2350json/'
    # json_dir = './new_json/'
    json_dir = '/media/trunk/sata/datasets/TRUNK/0802/gt/json/json_lidar/点云融合新规则交付json7304/20220802-7304'
    json_files = getFileList(json_dir, [], ext='json')
    trans = labelTranslator(K, extri_param)
    for json_file in json_files:
        cvtjson2txt(json_file, trans, out_dir, intra_coef)
