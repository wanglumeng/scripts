# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 23:24:09 2021

@author: aa
"""

import cv2
import numpy as np
import math

class ImDistortion:
    def __init__(self, camera_intra_coef, w, h):
        self.camera_intra_coef = camera_intra_coef
        self.w = w
        self.h = h
        self.F = np.array([[camera_intra_coef[0], 0, camera_intra_coef[2]], [0, camera_intra_coef[1], camera_intra_coef[3]], [0,0,1]])
        self.F_new, _ = cv2.getOptimalNewCameraMatrix(self.F, np.array(camera_intra_coef[4:]), (w, h), 1, (w, h))
        self.mapx_new, self.mapy_new = cv2.initUndistortRectifyMap(self.F, np.array(camera_intra_coef[4:]), None, self.F_new, (w, h), 5)
        self.mapx, self.mapy = cv2.initUndistortRectifyMap(self.F, np.array(camera_intra_coef[4:]), None, self.F, (w, h), 5)

    def DistortPoints(self, src, F_):
        fx = F_[0][0]
        fy = F_[1][1]
        ux = F_[0][2]
        uy = F_[1][2]

        k1 = self.camera_intra_coef[4]
        k2 = self.camera_intra_coef[5]
        p1 = self.camera_intra_coef[6]
        p2 = self.camera_intra_coef[7]
        k3 = self.camera_intra_coef[8]
        
        dst = src
        i = 0
        for p in src:
            px = (p[0] - ux)/fx
            py = (p[1] - uy)/fy
            r2 = px*px + py*py
            dA = 1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2
            dX = 2*p1*px*py + p2*(r2 + 2*px*px)
            dY = p1*(r2 + 2*py*py) +2*p2*px*py;
            xC = px * dA + dX
            yC = py * dA + dY

            xC = self.camera_intra_coef[0] * xC + self.camera_intra_coef[2]
            yC = self.camera_intra_coef[1] * yC + self.camera_intra_coef[3]
            dst[i] = [xC, yC]
            i = i+1
        return dst

    def undistortPts(self, src, F_):
        src_points_dt = cv2.undistortPoints(src, self.F,  np.array(self.camera_intra_coef[4:]), None, F_)    
        src_points_dt = src_points_dt[:,0,:]
        return src_points_dt

    def undistortImage(self, im, mapx_, mapy_):
        im = cv2.remap(im, mapx_, mapy_, cv2.INTER_LINEAR)
        return im



if __name__ == "__main__":
    fcameraMatrix = [1361.619212393047, 1362.499788592409, 660.8204792553835, 244.3957104029807, 
                  -0.5993876749576975, 0.4204803723239471, 0.01705854301089264, 0.003017995559993518, -0.3618606861029759]
    w = 1280
    h = 720
    imdt = ImDistortion(fcameraMatrix, w, h)

    im = cv2.imread('/home/fly/Downloads/data/r3d/data/yundong/301/out/camera_f60/time_1646102299550349824_1646102299506227493.jpg')
    cv2.imshow("org", im)
    im1 = imdt.undistortImage(im, imdt.mapx, imdt.mapy)
    cv2.imshow("und", im1)
    im2 = imdt.undistortImage(im, imdt.mapx_new, imdt.mapy_new)
    cv2.imshow("und2", im2)
    cv2.waitKey(0)
    pass


