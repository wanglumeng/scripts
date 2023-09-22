import numpy as np
import cv2


class BoxProcessor(object):
    '''
    img3d based bbox processor
    进行3D框的可视化，转换等
    '''
    def __init__(self, img3d_instance):
        self._img3d = img3d_instance
        self._dimension = {
            "car": [4, 2, 1.8],
            "truck": [20, 3, 4]
        }

    @staticmethod
    def roty(t):
        """ Rotation about the y-axis. """
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    @staticmethod
    def out_of_img(img_shape, pt):
        if pt[0] > img_shape[0] or pt[0] < 0 or pt[1] > img_shape[1] or pt[1] < 0:
            return 1
        return 0

    def convert_aabb_to_box(self, box_res, cls='car', dimension=None):
        if dimension:
            dimen_x = dimension[0]
            dimen_y = dimension[1]
            dimen_z = dimension[2]
        else:
            dimen_x = self._dimension[cls][0]
            dimen_y = self._dimension[cls][1]
            dimen_z = self._dimension[cls][2]
        center_x = box_res['long_pos']
        center_y = box_res['lat_pos']
        center_z = dimen_z / 2.0

        # car coord
        heading_yaw = 0  # rotate z
        heading_pitch = 0  # rotate y
        heading_roll = 0  # rotate x

        return [center_x, center_y, center_z, dimen_x, dimen_y, dimen_z,
                heading_yaw, heading_pitch, heading_roll]

    def project_r3d_to_image(self, img, r3d_info, draw_box=True, color=(0,255,0)):
        '''
        将car坐标系下的r3d投影到图像上，并且绘制在输入图像上进行可视化
        :param img:
        :param r3d_info:
        :param draw_box:
        :param color:
        :return:  可视化bbox的图像
        '''
        # compute car coord vertexes
        vertexes_car = self.compute_r3d_vertexes(r3d_info)

        vertexes_cam = self._img3d.car2cam(vertexes_car)
        img_arr, valid_mask = self._img3d.cam2img(vertexes_cam)

        # draw 3d box on image
        return self.draw_projected_box3d(img, img_arr, color)

    def compute_r3d_vertexes(self, r3d_info):
        '''
        计算顶点坐标: car 坐标系下
        :param r3d_info: [center_x, center_y, center_z, dimension_x, dimension_y, dimension_z, alpha]
        :return: car坐标系下的8个顶点坐标
        '''

        assert len(r3d_info) >= 7, "r3d box info length < 7, you need check it"

        # compute rotational matrix around yaw axis
        # todo cjs check the rotation
        R = self.roty(r3d_info[6])
        # 3d bounding box dimension
        l, w, h = r3d_info[3:6]

        # 3d bounding box corners
        x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
        y_corners = [-w/2, w/2, w/2, -w/2, -w/2, w/2, w/2, -w/2]
        z_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]

        # rotate and translate 3d bounding box
        corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
        # print corners_3d.shape
        corners_3d[0, :] = corners_3d[0, :] + r3d_info[0]
        corners_3d[1, :] = corners_3d[1, :] + r3d_info[1]
        corners_3d[2, :] = corners_3d[2, :] + r3d_info[2]

        return np.transpose(corners_3d)

    def box_valid_check(self, img_shape, qs):
        out_num = 0
        for k in range(0, 4):
            i, j = k, (k + 1) % 4
            out_num += self.out_of_img(img_shape, (qs[i, 0], qs[i, 1]))
            out_num += self.out_of_img(img_shape, (qs[j, 0], qs[j, 1]))
            i, j = k + 4, (k + 1) % 4 + 4
            out_num += self.out_of_img(img_shape, (qs[i, 0], qs[i, 1]))
            out_num += self.out_of_img(img_shape, (qs[j, 0], qs[j, 1]))
            i, j = k, k + 4
            out_num += self.out_of_img(img_shape, (qs[i, 0], qs[i, 1]))
            out_num += self.out_of_img(img_shape, (qs[j, 0], qs[j, 1]))
        return out_num < 8

    def draw_projected_box3d(self, image, qs, color=(0, 255, 0), thickness=2):
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
        # check whether the box is valid
        if not self.box_valid_check(image.shape, qs):
            return image

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

        # # fill
        # # img1 = np.copy(image)
        # img1 = np.zeros(image.shape, np.uint8)
        # img1[:, :, 0] = 255
        # pts = np.array([[[qs[0, 0], qs[0, 1]], [qs[1, 0], qs[1, 1]], [qs[5, 0], qs[5, 1]], [qs[4, 0], qs[4, 1]]]],
        #                dtype=np.int32)  # front
        # cv2.fillPoly(img1, pts, color)
        # image = cv2.addWeighted(img1, 0.3, image, 0.7, 1)  # transparency
        # cv2.line(image, [qs[0, 0], qs[0, 1]], [qs[5, 0], qs[5, 1]], color, thickness)
        #
        # blk = np.zeros(image.shape, np.uint8)
        # cv2.rectangle(blk, [qs[0, 0], qs[0, 1]], [qs[5, 0], qs[5, 1]], (255, 0, 0), -1)  # 注意在 blk的基础上进行绘制；
        # image = cv2.addWeighted(image, 1.0, blk, 0.5, 1)

        return image
