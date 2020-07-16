import cv2 as cv
import numpy as np
from collections import deque
import math
from computer_vision.detector_setup import detection_setup

class computer_vision():
    def __init__(self, render, quad_model, cv_cam, cv_cam_2, camera_cal1, camera_cal2):
        
        self.mtx1 = camera_cal1.mtx
        self.dist1 = camera_cal1.dist
        self.mtx2 = camera_cal2.mtx
        self.dist2 = camera_cal2.dist
        # print("Camera Matrix 1:",self.mtx1)
        # print("Distortion 1:", self.dist1)
        # print("3D point 1:", self.objpoint1)
        # print("Camera Matrix 2:",self.mtx2)
        # print("Distortion 2:", self.dist2)
        # print("3D point 2:", self.objpoint2)

        self.fast, self.criteria, self.nCornersCols, self.nCornersRows, self.objp, self.checker_scale, self.checker_sqr_size = detection_setup(render)

        self.render = render  
                
        self.render.quad_model.setPos(0, 0, 0)
        self.render.quad_model.setHpr(0, 0, 0)
        
        self.cv_cam = cv_cam
        self.cv_cam.cam.setPos(-0.2, 0, 6.2)
        self.cv_cam.cam.setHpr(0, 270, 0)
        self.cv_cam.cam.reparentTo(self.render.render)
        
        
        # self.cv_cam_2 = cv_cam_2
        # self.cv_cam_2.cam.setPos(0, 0, 6.5)
        # self.cv_cam_2.cam.setHpr(0, 270, 0)
        # #self.cv_cam_2.cam.lookAt(0, 0, 0)
        # self.cv_cam_2.cam.reparentTo(self.render.render)

        self.render.taskMgr.add(self.img_show, 'OpenCv Image Show')
    

    def detect_contourn(self, image):

        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        #Define the limits in HSV variables
        lower = np.array([0, 239, 222])
        upper = np.array([179, 255, 255])
        #Define threshold for red color
        mask = cv.inRange(hsv, lower, upper)
        #Create a kernel
        kernel = np.ones((5,5), np.uint8)
        #Apply opening process
        opening = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations = 1)
        #Find BLOB's contours
        _, cnts, _ = cv.findContours(opening.copy(), cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
             
        return  cnts

    def center_mass_calculate(self, image, c):
        # Compute the center of the contour
        M = cv.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        perimeter = cv.arcLength(c, True)
        #Compute the eccentricity
        metric = (4*math.pi*M["m00"])/perimeter**2
        if metric > 0.7:
            #Draw the contour and center of the shape on the image
            cv.drawContours(image, [c], -1, (255, 0, 0), 1)
            cv.circle(image, (cX, cY), 1, (255, 0, 0), 1)
        return cX, cY, metric


    def detect_corners(self, ret, image):

        if ret:
            img = cv.cvtColor(image, cv.COLOR_RGBA2BGR)
            self.gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)              
            ret, corners = cv.findChessboardCorners(self.gray, (self.nCornersCols, self.nCornersRows), 
                                                        cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FILTER_QUADS+ cv.CALIB_CB_FAST_CHECK)
            
            return ret, corners

    def draw(self, image, corners, imgpts):
        corner = tuple(corners[0].ravel())
        img = cv.line(image, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
        img = cv.line(image, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
        img = cv.line(image, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
        return img
    

    def get_pose(self, image, objpoints, imgpoints, mtx, dist):
        
        axis = np.float32([[.1,0,0], [0,.1,0], [0,0,0.1]]).reshape(-1,3)

        # Find the rotation and translation vectors.
        ret, rvecs, tvecs = cv.solvePnP(objpoints, imgpoints, mtx, dist)
        # project 3D points to image plane
        imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
        imgpts = imgpts.astype(np.int)
        img = self.draw(image, imgpoints, imgpts)
        
        return rvecs, tvecs, image


    def img_show(self, task):

        cX = None
        cY = None
        cX2 = None
        cY2 = None

        #Font setup
        font = cv.FONT_HERSHEY_PLAIN

        if task.frame % self.cv_cam.frame_int == 1:           
            ret, image = self.cv_cam.get_image()
            # ret, image2 = self.cv_cam_2.get_image()
            if ret:
                ret, corners = self.detect_corners(ret, image)
                if ret:                
                    if len(corners)==54:
                        rvecs, tvecs, image = self.get_pose(image, self.objp, corners, self.mtx1, self.dist1)

                        cv.putText(image, "X:"+str(tvecs[0])+"Y:"+str(tvecs[1])+ "Z:"+str(tvecs[2])
                        , (10,10), font, 1, (255, 255, 255), 1)
                # cnts = self.detect_contourn(image)
                # cnts2 = self.detect_contourn(image2)
                # loop over the contours
                # for c in cnts:
                #     cX, cY, _ = self.center_mass_calculate(image, c)
                # for c in cnts2:
                #     cX2, cY2, _ = self.center_mass_calculate(image2, c)

                #Print the image coordinates on the screen
                # cv.putText(image," Center:"+str(cX)+','+str(cY), (10, 10), font, 1, (255,255,255), 1)
                # cv.putText(image2," Center:"+str(cX2)+','+str(cY2), (10, 10), font, 1, (255,255,255), 1)

            cv.imshow('Drone Camera',image)
                # cv.imshow('Drone Camera 2 ',image2)
            cv.waitKey(1)

        return task.cont