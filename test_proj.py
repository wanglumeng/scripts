import cv2

from box_utils import BoxProcessor
from img3d import IMG3D
import json
import numpy as np


def draw_img_3dbox(canvas, color, track_loc, img3d_instance):
    # format track loc info
    b_p = BoxProcessor(img3d_instance)
    canvas = b_p.project_r3d_to_image(canvas, b_p.convert_aabb_to_box({
        'long_pos': track_loc[1], 'lat_pos': track_loc[2], 'height': track_loc[3]}, dimension=track_loc[8]), color=color)
    return canvas


if __name__ == "__main__":
    # img3d
    img3d_instance = IMG3D(json.load(open("./f60_cam.json")))
    img = np.ones((720, 1280, 3), dtype=np.uint8)*255
    box = [0, 100, 4, 1.5, 0, 0, 0, 0, [4, 2, 1], 0]
    box1 = [0, 50, 6, 1.5, 0, 0, 0, 0, [4, 2, 1], 0]
    box2 = [0, 20, 0, 1.5, 0, 0, 0, 0, [8, 2, 1], 0]
    box3 = [0, 30, 6, 1.5, 0, 0, 0, 0, [4, 2, 1], 0]
    box4 = [0, 80, 0, 1.5, 0, 0, 0, 0, [8, 2, 1], 0]
    draw_img_3dbox(img, (255, 0, 255), box, img3d_instance)
    draw_img_3dbox(img, (255, 0, 255), box1, img3d_instance)
    draw_img_3dbox(img, (255, 0, 255), box2, img3d_instance)
    draw_img_3dbox(img, (255, 0, 255), box3, img3d_instance)
    draw_img_3dbox(img, (255, 0, 255), box4, img3d_instance)

    cv2.imshow("res", img)
    cv2.waitKey(0)
