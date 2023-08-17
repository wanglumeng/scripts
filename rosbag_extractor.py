#-*-coding:utf-8-*-
#!/usr/bin/env python3
import roslib
#roslib.load_manifest('rosbag')
import rospy
import rosbag
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
import cv2
import numpy as np
import os
import json
import pickle


def header_format(msg_header):
    seq = int(msg_header.seq)
    stamp = int(msg_header.stamp.secs * 1e9 + msg_header.stamp.nsecs)
    frame_id = msg_header.frame_id

    return {"seq":seq, "stamp":stamp, "frame_id":frame_id}


class BagExtractor(object):
    def __init__(self, topic_name, ):
        self._topic_name = topic_name

    def extract(self, bag_path):
        print("=== Processing: {} ===\n"
              "--- Extracting: {} ---".format(bag_path, self._topic_name))
        bag = rosbag.Bag(bag_path, "r")
        for msg_info in bag.read_messages(self._topic_name):
            self.msg_process(msg_info)
        print("--- Finished ---\n")

    def msg_process(self, msg_info):
        raise NotImplementedError


class ImgBagExtractor(BagExtractor):
    def __init__(self, topic_name, save_dir, visualize=False):
        super(ImgBagExtractor, self).__init__(topic_name)
        self._save_dir = os.path.join(save_dir, topic_name.replace("/", "_").strip("_"))
        if not os.path.exists(self._save_dir):
            os.makedirs(self._save_dir)
        self._visualize = visualize

    def msg_process(self, msg_info):
        topic, msg, t = msg_info
        msg_header = msg.header
        topic_time = int(msg_header.stamp.secs * 1e9) + int(msg_header.stamp.nsecs)

        if (msg.format.find("rgb") != -1 or msg.format.find("bgr") != -1 or
            msg.format.find("bgra") != -1) or msg.format.find("jpg") != -1:
            np_arr = np.fromstring(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, -1)
            self.img2save(cv_image, topic_time)

    def img2save(self, img_data, t, visualize=False):
        out_path = os.path.join(self._save_dir, str(t) + ".jpg")
        # img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
        cv2.imwrite(out_path, img_data)
        if self._visualize:
            cv2.imshow("img", img_data)
            cv2.waitKey(10)


class RadarExtractor(BagExtractor):
    def __init__(self, topic_name, save_dir):
        super(RadarExtractor, self).__init__(topic_name)
        self._save_dir = os.path.join(save_dir, topic_name.replace("/", "_").strip("_"))
        if not os.path.exists(self._save_dir):
            os.makedirs(self._save_dir)

    def msg_process(self, msg_info):
        topic, msg, t = msg_info

        header_info = header_format(msg.header)
        frame_header = self.frame_header_process(msg)
        frame_content = self.frame_content_process(msg)

        out_content = {
            "msg_headar":header_info,
            "frame_header":frame_header,
            "frame_content":frame_content
        }
        self.save(out_content, os.path.join(self._save_dir, str(frame_header["stamp"])+".json"))

    def frame_header_process(self, frame):
        raise NotImplementedError

    def frame_content_process(self, frame):
        raise NotImplementedError

    def save(self, data, save_path):
        with open(save_path, "w") as save_f:
            json_content = json.dumps(data, indent=4)
            save_f.write(json_content)


class FrontRadarExtractor(RadarExtractor):
    def __init__(self, topic_name, save_dir):
        super(FrontRadarExtractor, self).__init__(topic_name, save_dir)

    def frame_header_process(self, frame):
        frame_header = frame.Header
        frame_info = {}
        frame_info["stamp"] = frame_header.Stamp
        # frame_info["stamp"] = int(frame.header.stamp.secs*1e9+frame.header.stamp.nsecs)
        frame_info["type"] = "front"
        frame_info["obj_num"] = frame_header.ObjNum
        frame_info["func_status"] = frame_header.FuncStatus
        frame_info["blockage"] = frame_header.Blockage
        frame_info["bus_off"] = frame_header.BusOff
        return frame_info

    def frame_content_process(self, frame):
        frame_content = frame.Data
        frame_res = []
        for one_obj in frame_content:
            obj_info = {}
            obj_info["ID"] = one_obj.ID
            obj_info["Type"] = one_obj.Type__
            obj_info["MoveStatus"] = one_obj.MoveStatus
            obj_info["Dx"] = one_obj.Dx + 5.36 + 5/2.   # todo tmp length
            obj_info["Dy"] = one_obj.Dy
            obj_info["Vx"] = one_obj.Vx
            obj_info["Vy"] = one_obj.Vy
            # TODO other info
            frame_res.append(obj_info)
        return frame_res


class SideRadarExtractor(RadarExtractor):
    def __init__(self, topic_name, save_dir):
        super(SideRadarExtractor, self).__init__(topic_name, save_dir)

    def msg_process(self, msg_info):
        topic, msg, t = msg_info

        header_info = header_format(msg.header)
        frame_header = self.frame_header_process(msg)
        frame_content = self.frame_content_process(msg)

        out_content = {
            "msg_headar":header_info,
            "frame_header":frame_header,
            "frame_content":frame_content
        }
        self.save(out_content, os.path.join(self._save_dir, str(frame_header["stamp"])+".json"))

    def frame_header_process(self, frame):
        frame_info = {}
        frame_header = frame.header
        frame_info["stamp"] = int(frame_header.stamp.secs * 1e9 + frame_header.stamp.nsecs)
        frame_info["type"] = "side"

        return frame_info

    def frame_content_process(self, frame):
        frame_content = frame.objects
        frame_res = []
        for one_obj in frame_content:
            if int(one_obj.sensor_obj_obj_class) == 0:
                continue
            obj_info = {}
            obj_info["ID"] = one_obj.sensor_obj_obj_id
            # obj_info["life_time"] = one_obj.sensor_obj_life_time
            # obj_info["obj_dyn_class"] = one_obj.sensor_obj_obj_dyn_class
            obj_info["Type"] = one_obj.sensor_obj_obj_class
            # obj_info["heading"] = one_obj.sensor_obj_heading
            length = (abs(one_obj.sensor_obj_long_ext_front) + abs(one_obj.sensor_obj_long_ext_back))
            length = 5 if length > 1e-1 else length
            obj_info["Dx"] = one_obj.sensor_obj_long_pos + 5.48 + length/2.  # to obj center
            obj_info["Vx"] = one_obj.sensor_obj_long_vel
            obj_info["Dy"] = one_obj.sensor_obj_lat_pos
            obj_info["Vy"] = one_obj.sensor_obj_lat_vel
            obj_info["MoveStatus"] = 1

            frame_res.append(obj_info)
        return frame_res


class OdomExtractor(BagExtractor):
    def __init__(self, topic_name, save_dir):
        super(OdomExtractor, self).__init__(topic_name)
        self._save_dir = os.path.join(save_dir, topic_name.replace("/", "_").strip("_"))
        if not os.path.exists(self._save_dir):
            os.makedirs(self._save_dir)
        self._cache_dict = {}

    def msg_process(self, msg_info):
        topic, msg, t = msg_info

        header_info = header_format(msg.header)
        frame_content = self.frame_content_process(msg)

        out_content = {
            "msg_headar":header_info,
            "frame_content":frame_content
        }

        self.save(frame_content, os.path.join(self._save_dir, str(header_info["stamp"])+".pkl"))

    def frame_content_process(self, frame):
        odom_info = {}
        odom_info["x"] = frame.pose.pose.position.x
        odom_info["y"] = frame.pose.pose.position.y
        odom_info["z"] = frame.pose.pose.position.z
        odom_info["qw"] = frame.pose.pose.orientation.w
        odom_info["qx"] = frame.pose.pose.orientation.x
        odom_info["qy"] = frame.pose.pose.orientation.y
        odom_info["qz"] = frame.pose.pose.orientation.z
        return odom_info

    def save(self, data, save_path):
        with open(save_path, "wb") as save_f:
            pickle.dump(data, save_f)


class VehicleStateExtractor(BagExtractor):
    def __init__(self, topic_name, save_dir):
        super(VehicleStateExtractor, self).__init__(topic_name)
        self._save_dir = os.path.join(save_dir, topic_name.replace("/", "_").strip("_"))
        if not os.path.exists(self._save_dir):
            os.makedirs(self._save_dir)
        self._cache_dict = {}

    def msg_process(self, msg_info):
        topic, msg, t = msg_info

        header_info = header_format(msg.header)
        frame_content = self.frame_content_process(msg)

        out_content = {
            "msg_headar":header_info,
            "frame_content":frame_content
        }

        self.save(out_content, os.path.join(self._save_dir, str(header_info["stamp"])+".json"))

    def frame_content_process(self, frame):
        pose_odom = {
            'x': frame.pose.translation.x,
            'y': frame.pose.translation.y,
            'z': frame.pose.translation.z,
            'qx': frame.pose.rotation.x,
            'qy': frame.pose.rotation.y,
            'qz': frame.pose.rotation.z,
            'qw': frame.pose.rotation.w
        }
        return pose_odom

    def save(self, data, save_path):
        with open(save_path, "w") as save_f:
            json_content = json.dumps(data, indent=4)
            save_f.write(json_content)


class StaticMapExtractor(BagExtractor):
    def __init__(self, topic_name, save_dir):
        super(StaticMapExtractor, self).__init__(topic_name)
        self._save_dir = os.path.join(save_dir, topic_name.replace("/", "_").strip("_"))
        if not os.path.exists(self._save_dir):
            os.makedirs(self._save_dir)
        self._cache_dict = {}

    def msg_process(self, msg_info):
        topic, msg, t = msg_info

        header_info = header_format(msg.header)
        frame_content = self.frame_content_process(msg)

        out_content = {
            "msg_headar":header_info,
            "frame_content":frame_content
        }

        self.save(out_content, os.path.join(self._save_dir, str(header_info["stamp"])+".json"))

    def frame_content_process(self, frame):
        map_info = {}
        map_info['sections']=[]
        for section in frame.sections:
            section_info = {}
            section_info["id"] = int(section.id)
            section_info['lanes'] = []
            for lane in section.lanes:
                lane_info= {}
                lane_info['id'] = lane.id
                lane_info['predecessors'] = lane.predecessors
                lane_info['successors'] = lane.successors
                lane_info['pts_left'] = [[pt.point.x, pt.point.y, pt.point.z] for pt in lane.pts_left]
                lane_info['pts_right'] = [[pt.point.x, pt.point.y, pt.point.z] for pt in lane.pts_right]
                lane_info['type'] = lane.type
                lane_info['turn_type'] = lane.turn_type
                lane_info['left_boundary_type'] = lane.left_boundary_type
                lane_info['right_boundary_type'] = lane.right_boundary_type
                section_info['lanes'].append(lane_info)
            section_info['predecessors'] = section.predecessors
            section_info['successors'] = section.successors
            map_info['sections'].append(section_info)
        return map_info

    def save(self, data, save_path):
        with open(save_path, "w") as save_f:
            json_content = json.dumps(data, indent=4)
            save_f.write(json_content)


class CurrentRegionExtractor(BagExtractor):
    def __init__(self, topic_name, save_dir):
        super(CurrentRegionExtractor, self).__init__(topic_name)
        self._save_dir = os.path.join(save_dir, topic_name.replace("/", "_").strip("_"))
        if not os.path.exists(self._save_dir):
            os.makedirs(self._save_dir)
        self._cache_dict = {}

    def msg_process(self, msg_info):
        topic, msg, t = msg_info

        header_info = header_format(msg.header)
        frame_content = self.frame_content_process(msg)

        out_content = {
            "msg_headar":header_info,
            "frame_content":frame_content
        }

        self.save(out_content, os.path.join(self._save_dir, str(header_info["stamp"])+".json"))

    def frame_content_process(self, frame):
        current_region = {}
        current_region['type']=frame.type
        current_region['turn_type']=frame.turn_type
        current_region['is_lockzone_ahead']=frame.is_lockzone_ahead

        return current_region

    def save(self, data, save_path):
        with open(save_path, "w") as save_f:
            json_content = json.dumps(data, indent=4)
            save_f.write(json_content)


class LidarExtractor(BagExtractor):
    def __init__(self, topic_name, save_dir):
        super(LidarExtractor, self).__init__(topic_name)
        self._save_dir = os.path.join(save_dir, topic_name.replace("/", "_").strip("_"))
        if not os.path.exists(self._save_dir):
            os.makedirs(self._save_dir)

    def msg_process(self, msg_info):
        topic, msg, t = msg_info

        frame_header = header_format(msg.header)
        frame_content = self.frame_content_process(msg)

        out_content = {
            "frame_header":frame_header,
            "frame_content":frame_content
        }
        self.save(out_content, os.path.join(self._save_dir, str(frame_header["stamp"])+".json"))

    def frame_content_process(self, frame):
        raise NotImplementedError

    def save(self, data, save_path):
        with open(save_path, "w") as save_f:
            json_content = json.dumps(data, indent=4)
            save_f.write(json_content)


class FrontLidarExtractor(LidarExtractor):
    def __init__(self, topic_name, save_dir):
        super(FrontLidarExtractor, self).__init__(topic_name, save_dir)

    def frame_content_process(self, frame):
        frame_content = frame.objects
        frame_res = []
        for one_obj in frame_content:
            obj_info = {}
            obj_info["ID"] = one_obj.id
            obj_info["Type"] = one_obj.classification
            obj_info["MoveStatus"] = one_obj.movingStatus
            obj_info["Dx"] = one_obj.Dx
            obj_info["Dy"] = one_obj.Dy
            # obj_info["Vx"] = one_obj.Vx
            # obj_info["Vy"] = one_obj.Vy
            obj_info["heading"] = one_obj.heading
            obj_info["length"] = one_obj.length
            obj_info["width"] = one_obj.width
            obj_info["height"] = one_obj.height
            # TODO other info
            frame_res.append(obj_info)
        return frame_res


class OldFrontLidarExtractor(LidarExtractor):
    def __init__(self, topic_name, save_dir):
        super(OldFrontLidarExtractor, self).__init__(topic_name, save_dir)

    def frame_content_process(self, frame):
        frame_content = frame.objects
        frame_res = []
        for one_obj in frame_content:
            obj_info = {}
            obj_info["ID"] = one_obj.id
            obj_info["Type"] = 1  # one_obj.classification
            obj_info["MoveStatus"] = 1  # one_obj.movingStatus
            obj_info["Dx"] = one_obj.object_box_center.x-one_obj.object_box_size.x/2.
            obj_info["Dy"] = one_obj.object_box_center.y
            obj_info["Vx"] = one_obj.relative_velocity.x
            obj_info["Vy"] = one_obj.relative_velocity.y
            obj_info["heading"] = one_obj.object_box_orientation  # rad clockwise-negative
            # TODO other info
            frame_res.append(obj_info)
        return frame_res


class CameraIntrinsicsExtractor(BagExtractor):
    """相机内参的存储

    """
    def __init__(self, topic_name, save_dir, visualize=False):
        super(CameraIntrinsicsExtractor, self).__init__(topic_name)
        self._save_dir = os.path.join(save_dir, topic_name.replace("/", "_").strip("_"))
        if not os.path.exists(self._save_dir):
            os.makedirs(self._save_dir)

    def msg_process(self, msg_info):
        topic, msg, t = msg_info
        msg_header = msg.header
        topic_time = int(msg_header.stamp.secs * 1e9) + int(msg_header.stamp.nsecs)

        # get intrinsics info
        camera_extrin_dict = {}
        camera_extrin_dict['IMG_H'] = msg.height
        camera_extrin_dict['IMG_W'] = msg.width
        camera_extrin_dict['fx'] = msg.K[0]
        camera_extrin_dict['cx'] = msg.K[2]
        camera_extrin_dict['fy'] = msg.K[4]
        camera_extrin_dict['cy'] = msg.K[5]
        camera_extrin_dict['coeff'] = msg.D
        
        self.intrin2save(camera_extrin_dict, topic_time, False)

    def intrin2save(self, info, t, visualize=False):
        out_path = os.path.join(self._save_dir, str(t) + ".json")
        out_json = json.dumps(info, indent=4)
        with open(out_path, "w") as out_f:
            out_f.write(out_json)

class CameraExtrinsicsExtractor(BagExtractor):
    """相机外参的存储(包括其他参考系的转换关系)

    """
    def __init__(self, topic_name, save_dir, visualize=False):
        super(CameraExtrinsicsExtractor, self).__init__(topic_name)
        self._save_dir = os.path.join(save_dir, topic_name.replace("/", "_").strip("_"))
        if not os.path.exists(self._save_dir):
            os.makedirs(self._save_dir)

    def msg_process(self, msg_info):
        res_dict = {}
        msg_time = 0
        # parse msg info 
        for tf_one in msg_info.message.transforms:
            tf_name, t, msg_dict = self.parse_extrinsics(tf_one)  
            msg_time = t
            res_dict[tf_name] = msg_dict
        # save msg to json
        self.extrin2save(res_dict, msg_time, False)

    def parse_extrinsics(self, msg_info):
        msg_header = msg_info.header
        msg_time = int(msg_header.stamp.secs * 1e9) + int(msg_header.stamp.nsecs)
        
        # get intrinsics info
        camera_extrin_dict = {}
        camera_extrin_dict['frame_id'] = msg_info.child_frame_id
        camera_extrin_dict['x'] = msg_info.transform.translation.x
        camera_extrin_dict['y'] = msg_info.transform.translation.y
        camera_extrin_dict['z'] = msg_info.transform.translation.z
        camera_extrin_dict['qw'] = msg_info.transform.rotation.w
        camera_extrin_dict['qx'] = msg_info.transform.rotation.x
        camera_extrin_dict['qy'] = msg_info.transform.rotation.y
        camera_extrin_dict['qz'] = msg_info.transform.rotation.z
        return msg_info.child_frame_id, msg_time, camera_extrin_dict
    
    def extrin2save(self, info, t, visualize=False):
        out_path = os.path.join(self._save_dir, str(t) + ".json")
        out_json = json.dumps(info, indent=4)
        with open(out_path, "w") as out_f:
            out_f.write(out_json)

topic_extractor_factory = {
        "/ARS430_input": FrontRadarExtractor,
        "/LFCr5tpRadarMsg": SideRadarExtractor,
        "/LRCr5tpRadarMsg": SideRadarExtractor,
        "/RFCr5tpRadarMsg": SideRadarExtractor,
        "/RRCr5tpRadarMsg": SideRadarExtractor,
        "/dev/video0/compressed": ImgBagExtractor,
        "/dev/video1/compressed": ImgBagExtractor,
        "/dev/video2/compressed": ImgBagExtractor,
        "/dev/video3/compressed": ImgBagExtractor,
        "/dev/video4/compressed": ImgBagExtractor,
        "/dev/video5/compressed": ImgBagExtractor,
        "/dev/video6/compressed": ImgBagExtractor,
        "/dev/video7/compressed": ImgBagExtractor,
        "/dev/video1_0/compressed": ImgBagExtractor,
        "/dev/video1_1/compressed": ImgBagExtractor,
        "/camera12_0/compressed": ImgBagExtractor,
        "/camera11_0/compressed": ImgBagExtractor,
        "/camera1_0/compressed": ImgBagExtractor,
        "/camera3_2/compressed": ImgBagExtractor,
        "/camera3_5/compressed": ImgBagExtractor,
        "/camera9_7/compressed": ImgBagExtractor,
        "/camera9_10/compressed": ImgBagExtractor,
        "/swift_odom": OdomExtractor,
        "/hadmap_server/static_map": StaticMapExtractor,
        "/pnc_msgs/vehicle_state": VehicleStateExtractor,
        "/hadmap_server/current_region": CurrentRegionExtractor,
        "/perception/objects": FrontLidarExtractor,
        "/camera/camera0_info": CameraIntrinsicsExtractor,
        "/camera/camera1_info": CameraIntrinsicsExtractor,
        "/camera/camera2_info": CameraIntrinsicsExtractor,
        "/camera/camera3_info": CameraIntrinsicsExtractor,
        "/camera/camera4_info": CameraIntrinsicsExtractor,
        "/camera/camera5_info": CameraIntrinsicsExtractor,
        "/camera/camera6_info": CameraIntrinsicsExtractor,
        "/tf_static": CameraExtrinsicsExtractor,
    }


if __name__ == "__main__":
    import sys
    assert len(sys.argv) == 4, "python rosbag_extractor.py topic_name dst_dir bag_name"
    topic_name = sys.argv[1]
    dst_dir = sys.argv[2]
    bag_name = sys.argv[3]

    extractor = topic_extractor_factory[topic_name](topic_name, dst_dir)

    if isinstance(bag_name, list):
        for bag_one in bag_name:
            extractor.extract(bag_one)
    else:
        extractor.extract(bag_name)

    