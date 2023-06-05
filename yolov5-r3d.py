import onnxruntime
import numpy as np
import math
import torch
import cv2
import torchvision
import time
import os

from tqdm import tqdm

PARAM_CAR_DICT = {
    '1219-f60-opt':{
        'In_Coef':((1304.28, 0, 657.276),
                    (0.0, 1303.43, 344.998),
                    (0.0, 0.0, 1.0)),
        'extri_param': {'yaw': 89.7, 'pitch': 178.721, 'roll': 83.25, 'x': 5.43, 'y': 0.001, 'z': 2.432},
        'DPs': (-0.588847, 0.462063, -0.00471503, -6.30203e-05,-0.4068)
    },
    '1219-l60-opt':{
        'In_Coef':((1338.95, 0, 642.469),
                    (0.0, 1340.33, 335.622),
                    (0.0, 0.0, 1.0)),
        'extri_param': {'yaw': 62.406, 'pitch': -2.005, 'roll': -97.047, 'x': 4.92, 'y': 1.49, 'z': 2.10},
        'DPs': (-0.60506, 0.428747, -0.00220089, 0.00253917, -0.208275)
    },
    '1219-r60-opt':{
        'In_Coef':((1317.83, 0, 605.598),
                    (0.0, 1320.98, 326.279),
                    (0.0, 0.0, 1.0)),
        'extri_param': {'yaw': 115.588, 'pitch': -3.438, 'roll': -96.30, 'x': 4.92, 'y': -1.49, 'z': 2.10},
        'DPs': (-0.541193, 0.197449, 0.000743007, -0.000560524, -0.131302)
    },
    '1219-f120-opt':{
        'In_Coef': ((658.5597414236421, 0, 635.1000935549321),
                    (0.0, 660.9405780565264, 346.8414113951293),
                    (0.0, 0.0, 1.0)),
        'extri_param': {'yaw': 88.84, 'pitch': 178.75, 'roll': 81.41, 'x': 4.98, 'y': 0.05, 'z': 2.36},
        'DPs': (-0.3025734483390898, 0.08524758316420078, -3.724102627785578e-05, -0.0005741611861387268, -0.01022103179680025)
    },
    '1219-l120-opt':{
        'In_Coef':((635.039, 0, 657.557),
                    (0.0, 635.478, 402.777),
                    (0.0, 0.0, 1.0)),
        'extri_param': {'yaw': 0.17, 'pitch': -2.29, 'roll': -105.47, 'x': 5.347, 'y': 1.50, 'z': 2.10},
        'DPs': (-0.341442, 0.138107, -0.000966168, 0.000174808, -0.0280139)
    },
    '1219-r120-opt':{
        'In_Coef':((640.359, 0, 641.432),
                    (0.0, 639.882, 385.316),
                    (0.0, 0.0, 1.0)),
        'extri_param': {'yaw': 178.396, 'pitch': -0.687, 'roll': -107.762, 'x': 4.95, 'y': -1.50, 'z': 2.10},
        'DPs': (-0.329275, 0.1237, -0.000313113, -0.000555863, -0.0225747)
    },
    
    '3670-f60-opt':{
        'In_Coef':((1304.28, 0, 657.276),
                    (0.0, 1303.43, 344.998),
                    (0.0, 0.0, 1.0)),
        'extri_param': {'yaw': 89.7, 'pitch': 178.721, 'roll': 83.25, 'x': 5.43, 'y': 0.001, 'z': 2.432},
        'DPs': (-0.588847, 0.462063, -0.00471503, -6.30203e-05,-0.4068)
    },
    '3670-l60-opt':{
        'In_Coef':((1302.28, 0, 630.631),
                    (0.0, 1301.83, 375.964),
                    (0.0, 0.0, 1.0)),
        'extri_param': {'yaw': 62.85, 'pitch': -1.29, 'roll': -96.29, 'x': 4.74, 'y': 1.47, 'z': 2.25},
        'DPs': (-0.571514, 0.314779, -0.00437051, 0.00106475, -0.0499429)
    },
    '3670-r60-opt':{
        'In_Coef':((1341.863, 0, 625.489),
                    (0.0, 1342.886, 325.606),
                    (0.0, 0.0, 1.0)),
        'extri_param': {'yaw': 117.36, 'pitch': -1.47, 'roll': -96.12, 'x': 5.18, 'y': -1.40, 'z': 2.09},
        'DPs': (-0.5711538077336606, 0.1625121491015549, 0.007248757517032669, 0.0008264400808089582, 0.3087890940139743)
    },
    '3670-f120-opt':{
        'In_Coef': ((658.5597414236421, 0, 635.1000935549321),
                    (0.0, 660.9405780565264, 346.8414113951293),
                    (0.0, 0.0, 1.0)),
        'extri_param': {'yaw': 88.84, 'pitch': 178.75, 'roll': 81.41, 'x': 4.98, 'y': 0.05, 'z': 2.36},
        'DPs': (-0.3025734483390898, 0.08524758316420078, -3.724102627785578e-05, -0.0005741611861387268, -0.01022103179680025)
    },
    '3670-l120-opt':{
        'In_Coef':((636.275, 0, 632.117),
                    (0.0, 644.476, 376.509),
                    (0.0, 0.0, 1.0)),
        'extri_param': {'yaw': 179.5, 'pitch': -179.293, 'roll': 74.115, 'x': 4.77, 'y': 1.49, 'z': 2.24},
        'DPs': (-0.314137, 0.101699, 0.000783991, -0.00261065, -0.0148193)
    },
    '3670-r120-opt':{
        'In_Coef':((632.275, 0, 634.117),
                    (0.0, 646.573, 372.619),
                    (0.0, 0.0, 1.0)),
        'extri_param': {'yaw': 180, 'pitch': -1, 'roll': -106, 'x': 4.77, 'y': -1.50, 'z': 2.25},
        'DPs': (-0.304436, 0.091689, 0.000733592, -0.00271045, -0.0135173)
    },
}


def init_session(model_path):
    EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    sess = onnxruntime.InferenceSession(model_path, providers=EP_list)
    return sess


# This is a wrapper to make the current InferenceSession class pickable.
class PickableInferenceSession:
    def __init__(self, model_path):
        self.model_path = model_path
        self.sess = init_session(self.model_path)

    def run(self, *args):
        return self.sess.run(*args)

    def __getstate__(self):
        return {'model_path': self.model_path}

    def __setstate__(self, values):
        self.model_path = values['model_path']
        self.sess = init_session(self.model_path)


class labelTranslator:
    def __init__(self, which_car, sensorl):
        self.camparams = {}
        for cam_name in sensorl:
            cam_para = f"{which_car}-{cam_name}-opt"
            param_cam = PARAM_CAR_DICT[cam_para]
            In_Coef = np.array(param_cam['In_Coef'])
            Proj = np.linalg.inv(In_Coef)
            DPs = np.array(param_cam['DPs'])
            Rt_cam2car, Rt_car2cam = self.extri_param2RT(param_cam['extri_param'])

            camparam = dict(
                In_Coef=torch.tensor(In_Coef).float(),
                Proj=torch.tensor(Proj).float(),
                DPs=torch.tensor(DPs).float(),
                Rt_cam2car=torch.tensor(Rt_cam2car).float(),
                Rt_car2cam=torch.tensor(Rt_car2cam).float(),
            )
            
            self.camparams[cam_name] = camparam
            
    def extri_param2RT(self, extri_param):
        rd = math.pi/180
        RT = np.array([[1,0,0,extri_param['x']],[0,1,0,extri_param['y']],[0,0,1,extri_param['z']],[0,0,0,1]])
        roll = extri_param['roll']*rd
        roll_mat = np.array([[1,0,0],[0,math.cos(roll), -math.sin(roll)], [0,math.sin(roll), math.cos(roll)]])
        pitch = extri_param['pitch']*rd
        pitch_mat = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0,1,0],[-math.sin(pitch), 0, math.cos(pitch)]])
        yaw = extri_param['yaw']*rd
        yaw_mat = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0],[0,0,1]])
        rot_mat = yaw_mat @ pitch_mat @ roll_mat
        RT[:-1, :-1] = rot_mat
        Rt_cam2car = RT
        Rt_car2cam = np.linalg.inv(RT)
        return Rt_cam2car, Rt_car2cam


def decode_location(points,
                    depths,
                    Proj):
    '''
    retrieve objects location in camera coordinate based on projected points
    Args:
        points: bbox center
        depths: object depth z

    Returns:
        locations: objects location, shape = [N, 3]
    '''

    # Proj = self.P64[:,:3].to(points.device)
    ones = torch.ones(points.shape[0], device=points.device).unsqueeze(-1)
    pts = torch.cat((points, ones), dim=-1)[..., None]

    rays = torch.matmul(Proj, pts).squeeze(-1)
    betas = torch.atan(rays[:, 0]/rays[:, 2])
    locations = depths.view(-1, 1) * rays

    return locations, betas


def cam2car(Rt, camloc, roty=None):
    Rt = Rt.to(camloc.device)
    if len(Rt.shape) == 2:
        Rt = Rt.unsqueeze(0)
    camloc = torch.cat((camloc, torch.ones_like(
        camloc[:, 0:1])), dim=-1)[..., None]
    carloc = torch.matmul(Rt, camloc).squeeze(-1)
    carloc3 = carloc[..., :-1]

    if isinstance(roty, torch.Tensor):
        rots = torch.zeros((roty.shape[0], 4, 1), device=roty.device)
        rots[:, 0, :] = torch.cos(roty)
        rots[:, 2, :] = -torch.sin(roty)
        ori_z = torch.matmul(Rt, rots).squeeze(-1)
        rotz = -torch.atan2(ori_z[:, 1], ori_z[:, 0])
        return carloc3, rotz
    else:
        return carloc3


def decode_R3dPred(predn,
                   camparam,
                   base_f=1300,
                   DECODE_DIS=False):
    device = predn.device
    In_Coef = camparam['In_Coef'].to(device)
    Proj = camparam['Proj'].to(device)
    Rt_cam2car = camparam['Rt_cam2car'].to(device)

    # pbox0, pconf0, pcls0, pyaw0, pdim0, pdis0, pwxa0, pwxb0, pwya0, pboff0 = predn.split((4, 1, 1, 2, 3, 1, 1, 1, 1, 1), -1)
    pbox0, pconf0, pcls0, pyaw0, pdim0, pdis0, pwxb0, pboff0 = predn.split(
        (4, 1, 1, 2, 3, 1, 2, 1), -1)

    pyaw0 = torch.atan2(pyaw0[..., 0], pyaw0[..., 1]).unsqueeze(-1)

    pdis0 = pdis0*In_Coef[0, 0]/base_f

    pcent = torch.stack(
        ((pbox0[:, 0] + pbox0[:, 2])/2., (pbox0[:, 1] + pbox0[:, 3])/2.), dim=-1)
    ploc, pbeta = decode_location(pcent, pdis0, Proj)

    if DECODE_DIS:
        proty = pyaw0 + pbeta.unsqueeze(-1)
        proty = torch.where(proty > math.pi, proty-2*math.pi, proty)
        proty = torch.where(proty < -math.pi, proty+2*math.pi, proty)
        pwx0, pwy0, pwz0 = ploc.split((1, 1, 1), -1)

    else:
        pbeta3D = pbeta[:, None]
        pwx0 = pdis0 * torch.cos(pbeta3D) + pwxb0[:, 0:1]
        pwy0 = pdis0 * torch.sin(pbeta3D) + pwxb0[:, 1:2]
        pwz0 = ploc[:, 1:2]

    predn_cam = torch.cat((pbox0, pconf0, pcls0, pyaw0,
                          pdim0, pwx0, pwy0, pwz0, pdis0), -1)

    # location transfer
    predn_car = predn_cam.clone()
    ploc_cam = predn_cam[:, [11, 12, 10]]
    proty = predn_cam[:, 6:7]
    ploc_car, protz = cam2car(Rt_cam2car, ploc_cam, proty)
    predn_car[:, 10:13] = ploc_car
    predn_car[:, 6] = protz

    return predn_cam, predn_car


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):

    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0],
                   img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
            img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    # self.clip_coords(coords, img0_shape)
    return coords


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - \
        new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / \
            shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def box_area(box):
    # box = xyxy(4,n)
    return (box[2] - box[0]) * (box[3] - box[1])


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1[:, None].chunk(2, 2), box2.chunk(2, 1)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / (box_area(box1.T)[:, None] + box_area(box2.T) - inter)


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def non_max_suppression(prediction,
                        conf_thres=0.25,
                        iou_thres=0.45,
                        classes=None,
                        agnostic=False,
                        multi_label=False,
                        labels=(),
                        max_det=300,
                        nc=None):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping bounding boxes

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    bs = prediction.shape[0]  # batch size
    if not nc:
        nc = prediction.shape[2] - 5 - 9  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.1 + 0.03 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:5+nc] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:5+nc] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None],
                          j[:, None].float(), x[i, nc + 5:]), 1)
        else:  # best class only
            conf, j = x[:, 5:5+nc].max(1, keepdim=True)
            x = torch.cat(
                (box, conf, j.float(), x[:, nc + 5:]), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            # sort by confidence
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        # boxes (offset by class), scores
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float(
            ) / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]

    return output


def load_model(weights='yolov5s.pt', device=torch.device('cpu')):
    w = str(weights[0] if isinstance(weights, list) else weights)
    print(f'Loading {w} for ONNX Runtime inference...')
    cuda = torch.cuda.is_available()
    providers = ['CUDAExecutionProvider',
                 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
    session = onnxruntime.InferenceSession(w, providers=providers)
    return session


class Yolov5R3D():
    def __init__(self, onnx_path, imgsz=(512, 512), sensorl=['f60'], which_car='3670', device="cuda"):
        """
        :param onnx_path:
        """
        self.device = device
        self.onnx_session = PickableInferenceSession(onnx_path)
        # self.session = load_model(onnx_path, device)
        self.tran = labelTranslator(which_car, sensorl)
        self.imgsz = imgsz
        self.sensorl = sensorl
        self.stride = 32
        self.ids_name = {0: "car", 1: "truck", 2: "bus"}
        self.c2d_name = {0: "cone", 1: "ped", 2: "cyc", 3: "tricar"}

    def forward(self, img_path):
        image = cv2.imread(img_path)
        fname = os.path.basename(img_path)
        sensor = fname.split('_')[0]
        
        if sensor not in self.sensorl:
            return
        
        camparam = self.tran.camparams[sensor]

        img = image.copy()
        img = letterbox(img, self.imgsz, stride=self.stride, auto=False)[0]
        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = img.astype(dtype=np.float32)
        img /= 255.  # 0 - 255 to 0.0 - 1.0
        img = np.expand_dims(img, axis=0)

        predss = self.onnx_session.run(None, {"images": img})
        
        if len(predss) == 4:
            pred, pred_r3d, predc, pred_c3d = predss
        if len(predss) == 2:
            pred, pred_r3d = predss
            
        # object_conf = conf_2d * dept_conf
        pred[..., 4] *= pred_r3d[..., -1]
        
        # pred, pred_r3d = self.session.run(None, {self.session.get_inputs()[0].name: img})
        pred = torch.cat((torch.from_numpy(pred).to(self.device),
                         torch.from_numpy(pred_r3d).to(self.device)), 2)
        # outc = torch.from_numpy(outc).to(self.device)
        
        conf_thres = 0.25
        iou_thres = 0.45
        agnostic_nms = False
        # pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        pred = non_max_suppression(
            pred, conf_thres, iou_thres, None, agnostic_nms)
        # c2d_out = non_max_suppression(outc, conf_thres, iou_thres, None, nc=4)
        
        output = []
        
        if len(pred[0]) == 0:
            if save_txt:
                save_empty_txt(file=out_path + fname.replace('jpg', 'txt')) 
        else:
            for i, det in enumerate(pred):  # per image
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    self.imgsz, det[:, :4], image.shape).round()

                predn_cam, predn_car = decode_R3dPred(
                    det, camparam, base_f=1300, DECODE_DIS=False)
                
                if save_txt:
                    save_one_txt(predn_car, True, image.shape, file=out_path + fname.replace('jpg', 'txt'))
                
                for r3d_info in predn_car:
                    r3d_info = r3d_info.cpu().numpy()
                    bboxes = r3d_info[0:4].tolist()
                    # cls = self.ids_name[int(r3d_info[5])]
                    cls = int(r3d_info[5])
                    conf = r3d_info[4]
                    roytwhlxyz = r3d_info[6:].tolist()
                    output.append(dict(
                                    box=bboxes,
                                    cls=cls,
                                    conf=conf,
                                    roys=roytwhlxyz[0],
                                    lwh=roytwhlxyz[1:4],
                                    xyz=roytwhlxyz[4:7]

                                    ))

        # for si, predc in enumerate(c2d_out):
        #     if len(predc):
        #         prednc = predc.clone()
        #         scale_coords(self.imgsz, prednc[:, :4], image.shape)  # native-space pred
                
        #         for c2d_info in prednc:
        #             c2d_info = c2d_info.cpu().numpy()
        #             bboxes = c2d_info[0:4].tolist()
        #             cls = self.c2d_name[int(c2d_info[5])]
        #             conf = c2d_info[4]
        #             output.append(dict(
        #                           box=bboxes,
        #                           cls=cls,
        #                           conf=conf,
        #                           ))
        return output

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

def save_one_txt(predn, save_conf, shape, file):
    pred_2d, pred_3d = predn[:, :6], predn[:, 6:-1]
    # Save one txt result
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for i, (*xyxy, conf, cls) in enumerate(pred_2d.tolist()):
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = [cls, *xywh] + pred_3d[i].tolist() + [conf] if save_conf else [cls, *xywh] + pred_3d[i].tolist() 
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % tuple(line) + '\n')
            
def save_empty_txt(file):
    with open(file, 'w') as f:
        f.write('\n')

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

def create_dir_not_exist(path, RM=False):
    import shutil
    if os.path.isdir(path):
        if RM:
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


if __name__ == '__main__':

    # onnx_path = "/home/fly/Projects/VSProjects/CppProjects/vision_ws/data/model/model/r3d-v5.4-c3d_v3.5_r3d-pgtc3d-conepedcyc2d_newcanch_bs96/weights/best_bs1.onnx"
    # onnx_path = "/home/fly/Downloads/data/r3d/output/model/r3d-v5.2-dconfv1_v3.5_r3d-ori_bs128_e50/weights/best.onnx"
    # onnx_path = "/home/fly/Downloads/data/r3d/output/model/r3d-v5.2-dconf-v2_v3.5_r3d-ori_bs128_e50_t/weights/best_bs1.onnx"
    # onnx_path = "/home/fly/Downloads/data/r3d/output/model/r3d-v5.2-dconfv3_v3.5_r3d-ori_bs64_e50/weights/best.onnx"
    onnx_path = "/home/fly/Projects/VSProjects/CppProjects/vision_ws/data/model/model/r3d-v5.2-dconf-v2_v3.5_r3d-ori_bs128_e50_t/weights/best_bs1.onnx"
    
    # 图片命名的方式需要前缀为相机名称如 f60_xxxxxxxxxxxx.jpg/l120_xxxxxxxxxxxxx.jpg
    data_path = "/home/fly/Projects/VSProjects/CppProjects/vision_ws/data/bag/issue/误检/VTI-7408_匝道内左侧视觉误识别/images/"
    out_path = "/home/fly/Projects/VSProjects/CppProjects/vision_ws/data/bag/issue/误检/VTI-7408_匝道内左侧视觉误识别/dconfv2/labels/"
    imgsz = (288, 512)
    which_car = '3670'
    sensorl = ["f60", "l60", "l120", "r60", "r120"]
    save_txt = False
    
    create_dir_not_exist(out_path, RM=False)
    yolov5 = Yolov5R3D(onnx_path, imgsz, sensorl, which_car, device="cuda")
    
    img_files = getFileList(data_path, [], ext='jpg')
    
    for img_path in tqdm(img_files, desc="inferencing image"):
        # print("\ninferencing image {}".format(img_path))
        output = yolov5.forward(img_path)
        
            

        
        # for out in output:
        #     bbox = list(map(int, out['box']))
        #     x1, y1, x2, y2 = bbox
        #     cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)),
        #                               color=(255, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        # cv2.imwrite("f60_res.jpg", img)
        # print(output)
