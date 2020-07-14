import cv2 as cv
import numpy as np
from collections import deque
import math


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
    def __init__(self, render, quad_model, quad_env, quad_sens, quad_pos, cv_cam, cv_cam_2, camera_cal, mydir, IMG_POS_DETER):
        
        self.mtx = camera_cal.mtx
        self.dist = camera_cal.dist
        
        self.IMG_POS_DETER = IMG_POS_DETER

        self.quad_env = quad_env
        self.quad_sens = quad_sens
        self.image_pos = None
        self.vel_sens = deque(maxlen=100)
        self.vel_img = deque(maxlen=100)
        self.render = render  
                

        self.render.quad_model.setPos(0, 0, 0)
        self.render.quad_model.setHpr(0, 0, 0)
        
        self.cv_cam = cv_cam
        self.cv_cam.cam.setPos(0, 0, 6.5)
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
        #Setup de fonte
        font = cv.FONT_HERSHEY_PLAIN

        if task.frame % 10 == 1:           
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
                    print("Metric:",metric)
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
                    print("Metric2:",metric)
                    if metric > 0.7:
        
                        #draw the contour and center of the shape on the image
                        cv.drawContours(image2, [c], -1, (255, 0, 0), 1)
                        cv.circle(image2, (cX2, cY2), 1, (255, 0, 0), 1)
    
    
                cv.putText(image," Center:"+str(cX)+','+str(cY), (10, 80), font, 1, (255,255,255), 1)
                cv.putText(image2," Center:"+str(cX2)+','+str(cY2), (10, 80), font, 1, (255,255,255), 1)

                cv.imshow('Drone Camera',image)
                cv.imshow('Drone Camera 2 ',image2)
                cv.waitKey(1)
        return task.cont