import cv2 as cv
import numpy as np
from collections import deque
import math
from computer_vision.detector_setup import detection_setup


# #Criar janela para trackbar
    # cv.namedWindow("Trackbars")

    # #Criar trackbars
    # cv.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
    # cv.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
    # cv.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
    # cv.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
    # cv.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
    # cv.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

class computer_vision():
    def __init__(self, render, quad_model, cv_cam, cv_cam_2, camera_cal1, camera_cal2):
        
        self.mtx1 = camera_cal1.mtx
        self.dist1 = camera_cal1.dist
        self.objpoint1 = camera_cal1.objp
        self.mtx2 = camera_cal2.mtx
        self.dist2 = camera_cal2.dist
        self.objpoint2 = camera_cal2.objp
        print("Camera Matrix 1:",self.mtx1)
        print("Distortion 1:", self.dist1)
        print("3D point 1:", self.objpoint1)
        print("Camera Matrix 2:",self.mtx2)
        print("Distortion 2:", self.dist2)
        print("3D point 2:", self.objpoint2)

        self.fast, self.criteria, self.nCornersCols, self.nCornersRows, self.objp, self.checker_scale, self.checker_sqr_size = detection_setup(render)

        self.render = render  
                
        self.render.quad_model.setPos(0, 0, 0)
        self.render.quad_model.setHpr(0, 0, 0)
        
        self.cv_cam = cv_cam
        self.cv_cam.cam.setPos(0.5, 0, 6.5)
        self.cv_cam.cam.setHpr(0, 270, 0)
        #self.cv_cam.cam.lookAt(0, 0, 0)
        self.cv_cam.cam.reparentTo(self.render.render)
        
        
        self.cv_cam_2 = cv_cam_2
        self.cv_cam_2.cam.setPos(0.1, 0, 6.5)
        self.cv_cam_2.cam.setHpr(0, 270, 0)
        #self.cv_cam_2.cam.lookAt(0, 0, 0)
        self.cv_cam_2.cam.reparentTo(self.render.render)

        self.render.taskMgr.add(self.img_show, 'OpenCv Image Show')
    
    

    def img_show(self, task):

        cX = None
        cY = None
        cX2 = None
        cY2 = None
        x1 = None
        y1 = None
        tvecs = np.array
        #Setup de fonte
        font = cv.FONT_HERSHEY_PLAIN

        if task.frame % self.cv_cam.frame_int == 1:           
            ret, image = self.cv_cam.get_image()
            ret, image2 = self.cv_cam_2.get_image()
            if ret:
                #Converte frame para HSV
                hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
                hsv2 = cv.cvtColor(image2, cv.COLOR_BGR2HSV)
                # l_h = cv.getTrackbarPos("L - H", "Trackbars")
                # l_s = cv.getTrackbarPos("L - S", "Trackbars")
                # l_v = cv.getTrackbarPos("L - V", "Trackbars")
                # u_h = cv.getTrackbarPos("U - H", "Trackbars")
                # u_s = cv.getTrackbarPos("U - S", "Trackbars")
                # u_v = cv.getTrackbarPos("U - V", "Trackbars")

                #Detecção de cor através de HSV
                # lower = np.array([l_h, l_s, l_v])
                # upper = np.array([u_h, u_s, u_v])
                lower = np.array([0, 239, 222])
                upper = np.array([179, 255, 255])
                #Cria mascara para filtrar o objeto pela cor definida pelos limites
                mask = cv.inRange(hsv, lower, upper)
                mask2 = cv.inRange(hsv2, lower, upper)
                #Cria kernel
                kernel = np.ones((5,5), np.uint8)
                #Aplica processo de Abertura (Erosão seguido de Dilatação)
                opening = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations = 1)
                opening2 = cv.morphologyEx(mask2, cv.MORPH_OPEN, kernel, iterations = 1)
    
    
                _, cnts, _ = cv.findContours(opening.copy(), cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
                _, cnts2, _ = cv.findContours(opening2.copy(), cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

                # loop over the contours
                for c in cnts:
                    # compute the center of the contour
                    M = cv.moments(c)
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    
                    perimeter = cv.arcLength(c, True)
                    metric = (4*math.pi*M["m00"])/perimeter**2
                    if metric > 0.7:
        
                        #draw the contour and center of the shape on the image
                        cv.drawContours(image, [c], -1, (255, 0, 0), 1)
                        cv.circle(image, (cX, cY), 1, (255, 0, 0), 1)
                
                for c in cnts2:
                    # compute the center of the contour
                    M = cv.moments(c)
                    cX2 = int(M["m10"] / M["m00"])
                    cY2 = int(M["m01"] / M["m00"])
                    
                    perimeter = cv.arcLength(c, True)
                    metric = (4*math.pi*M["m00"])/perimeter**2
                    if metric > 0.7:
        
                        #draw the contour and center of the shape on the image
                        cv.drawContours(image2, [c], -1, (255, 0, 0), 1)
                        cv.circle(image2, (cX2, cY2), 1, (255, 0, 0), 1)

                
                image_b = cv.cvtColor(image, cv.COLOR_RGBA2BGR)
                gray = cv.cvtColor(image_b, cv.COLOR_BGR2GRAY)
                fast_gray = cv.resize(gray, None, fx=1, fy=1)
                corner_good = self.fast.detect(fast_gray)
                if len(corner_good) > 83:
                    point = []
                    for kp in corner_good:
                        point.append(kp.pt)
                    point = np.array(point)
                    mean = np.mean(point, axis=0)
                    var = np.var(point, axis=0)
                    if var[0] < 30000 and var[1] < 10000:
                        ret, corners = cv.findChessboardCorners(image_b, (self.nCornersCols, self.nCornersRows),
                                                                cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE)
                        if ret:
                            # corners2 = cv.cornerSubPix(self.gray, corners, (1, 1), (-1, -1), self.criteria)
                            ret, rvecs, tvecs = cv.solvePnP(self.objp, corners, self.mtx1, self.dist1)
                            x1 = tvecs[0]
                            y1 = tvecs[1]

                cv.putText(image," Center:"+str(cX)+','+str(cY), (10, 10), font, 1, (255,255,255), 1)
                cv.putText(image," Real Center:"+str(x1)+','+str(y1), (10, 30), font, 1, (255,255,255), 1)
                
                cv.putText(image2," Center:"+str(cX2)+','+str(cY2), (10, 30), font, 1, (255,255,255), 1)

                cv.imshow('Drone Camera',image)
                #cv.imshow('Drone Camera 2 ',image2)
                cv.waitKey(1)
        return task.cont