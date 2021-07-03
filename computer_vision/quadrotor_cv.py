import cv2 as cv
import numpy as np
from numpy.linalg import inv, det
import scipy as sci
from scipy.spatial.transform import Rotation as R
from collections import deque
import math
import statistics
import time
import os
from matplotlib import pyplot as plt
from computer_vision.detector_setup import detection_setup
from computer_vision.img_2_cv import opencv_camera
from environment.quadrotor_env import sensor
from computer_vision.quat_utils import Quat2Rot, QuatProd, SkewMat, DerivQuat, Euler2Quat



class computer_vision():
    def __init__(self, render, quad_model, cv_cam, cv_cam_2, camera_cal1, camera_cal2, quad_position):
        


        self.mtx1 = camera_cal1.mtx
        self.dist1 = camera_cal1.dist

        print(self.mtx1)

        self.mtx2 = camera_cal2.mtx
        self.dist2 = camera_cal2.dist
        # print(self.dist1)


        self.render = render  
        self.quad_position = quad_position
        self.render.quad_model.setPos(0, 0, 0)
        self.render.quad_model.setHpr(0, 0, 0)
        
        #Camera Setup
        self.cv_cam = cv_cam
        self.cv_cam.cam.node().getLens().setFilmSize(36, 24)
        self.cv_cam.cam.node().getLens().setFocalLength(40)
        self.cv_cam.cam.setPos(0, -2.5, 6)
        self.cv_cam.cam.setHpr(0, 305, 0)
        # self.cv_cam.cam.lookAt(0, 0, 0)
        self.cv_cam.cam.reparentTo(self.render.render)

        self.cv_cam_2 = cv_cam_2
        self.cv_cam_2.cam.node().getLens().setFilmSize(36, 24)
        self.cv_cam_2.cam.node().getLens().setFocalLength(40)
        self.cv_cam_2.cam.setPos(0, 2.5, 6)
        self.cv_cam_2.cam.setHpr(180, 305, 0)
        # self.cv_cam.cam.lookAt(0, 0, 0)
        self.cv_cam_2.cam.reparentTo(self.render.render)


        #Some local variable
        self.index = 0
        self.index2 = 0
        self.time = 0

        self.position_mean = None
        self.cam_vec = None
        self.cam_vec2 = None

        self.cx_list = deque(maxlen=3)
        self.cy_list = deque(maxlen=3)
        self.cz_list = deque(maxlen=3)

        self.cx_list2 = deque(maxlen=3)
        self.cy_list2 = deque(maxlen=3)
        self.cz_list2 = deque(maxlen=3)

        self.mean_cx = None
        self.mean_cy = None
        self.mean_cz = None
        self.mean2_cx = None
        self.mean2_cy = None
        self.mean2_cz = None

        self.cx_list2 = []
        self.cy_list2 = []
        self.cz_list2 = []
        


        #Run Tasks
        # self.render.taskMgr.add(self.img_show, 'OpenCv Image Show')


    def aruco_detection(self, image, dictionary, rvecs, tvecs, cx_list, cy_list, cz_list, mean_cx, mean_cy, mean_cz):
        
        position = None
        q_obj_b = None
        cam_vec = None
        camera_meas_flag = None

        image = cv.cvtColor(image, cv.COLOR_BGRA2BGR)
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        #Initialize the detector parameters using defaults values
        parameters = cv.aruco.DetectorParameters_create()
        parameters.adaptiveThreshWinSizeMin = 5
        parameters.adaptiveThreshWinSizeMax = 20
        parameters.adaptiveThreshWinSizeStep = 3
        parameters.adaptiveThreshConstant = 20

        parameters.cornerRefinementWinSize = 15
        parameters.cornerRefinementMethod = cv.aruco.CORNER_REFINE_CONTOUR
        # parameters.cornerRefinementMaxIterations = 20
        parameters.cornerRefinementMinAccuracy = 0.01


        #Detect the markers in the image
        markerCorners, markerIDs, rejectedCandidates = cv.aruco.detectMarkers(gray, dictionary, parameters=parameters)
        
        

        #If there is a marker compute the pose
        if markerIDs is not None and len(markerIDs)==2:

            #Descending order Marker ID's array
            if len(markerIDs)==2:
                order = np.argsort(-markerIDs.reshape(1,2))
                markerIDs = markerIDs[order].reshape(2,1)
                markerCorners = np.asarray(markerCorners, dtype='float32')[order]

            for i in range(0, len(markerIDs)):
                
                #Check if it's reference marker
                if markerIDs[i]==10:
                    
                    #Compute Pose Estimation of the Reference Marker
                    rvec_ref, tvec_ref, _ = cv.aruco.estimatePoseSingleMarkers(markerCorners[0][i], 0.5, self.mtx1, self.dist1, rvecs, tvecs)
                    rvec_ref = np.reshape(rvec_ref, (3,1))
                    tvec_ref = np.reshape(tvec_ref, (3,1))
                    #Use Rodrigues formula to transform rotation vector into matrix
                    #Pose marker w.r.t camera reference frame
                    R_rc, _ = cv.Rodrigues(rvec_ref)
                    #Homogeneous Transformation Fixed Frame to Camera Frame
                    last_col = np.array([[0, 0, 0, 1]])
                    T_rc = np.concatenate((R_rc, tvec_ref), axis=1)
                    T_rc = np.concatenate((T_rc, last_col), axis=0)
                    #Homegeneous Transformation Camera Frame to Fixed Frame
                    T_cr = np.linalg.inv(T_rc)

                    #Get Reference's Marker attitude w.r.t camera
                    r_ref = sci.spatial.transform.Rotation.from_matrix(T_cr[0:3, 0:3])
                    q_ref = r_ref.as_quat()
                    euler_ref = r_ref.as_euler('XYZ')

                    #Draw axis in marker
                    cv.aruco.drawAxis(image, self.mtx1, self.dist1, rvec_ref, tvec_ref, 0.5)
                    cv.aruco.drawDetectedMarkers(image, markerCorners[0])

                #Check if there is moving marker/object marker
                if markerIDs[i]==4:
                

                    #Get Pose Estimation of the Moving Marker
                    rvec_obj, tvec_obj, _ = cv.aruco.estimatePoseSingleMarkers(markerCorners[0][i], 0.185, self.mtx1, self.dist1, rvecs, tvecs)
                    rvec_obj = np.reshape(rvec_obj, (3,1))
                    tvec_obj = np.reshape(tvec_obj, (3,1))

                    #Use Rodrigues formula to transform rotation vector into matrix
                    R_mc, _ = cv.Rodrigues(rvec_obj)

                    #Homogeneous Transformation Object Frame to Camera Frame
                    last_col = np.array([[0, 0, 0, 1]])
                    T_mc = np.concatenate((R_mc, tvec_obj), axis=1)
                    T_mc = np.concatenate((T_mc, last_col), axis=0)

                    r_mov = sci.spatial.transform.Rotation.from_matrix(T_mc[0:3, 0:3])
                    q_mov = r_mov.as_quat()
                    euler_mov = r_mov.as_euler('XYZ')
                    

                    if T_cr is not None:
                        #Homogeneous Transformation Object Frame to Fixed Frame
                        T_mr = T_cr@T_mc
                    else:
                        T_mr = np.eye(4)@T_mc

                    T_rm = T_mr.T

                    #Getting quaternions from rotation matrix
                    r_obj = sci.spatial.transform.Rotation.from_matrix(T_mr[0:3, 0:3])
                    q_obj = r_obj.as_quat()
                    euler_obj = r_obj.as_euler('XYZ')
                    
                    #Getting quaternion from Fixed Frame to Body Frame
                    r_obj_b = sci.spatial.transform.Rotation.from_matrix(T_rm[0:3, 0:3])
                    r_real = self.quad_position.quad_env.state[6:10]
                    q_obj_b = r_obj_b.as_quat()


                    #Marker's Position
                    zf_obj = float(T_mr[2,3])
                    xf_obj = float(T_mr[0,3])
                    yf_obj = float(T_mr[1,3])

                    position = np.array([[xf_obj, yf_obj, zf_obj]]).T


                    #Measurement Direction Vector
                    
            
                    cx = float(q_obj_b[3])**2 + float(q_obj_b[0])**2 - float(q_obj_b[1])**2 - float(q_obj_b[2])**2
                    cy = 2*(float(q_obj_b[0])*float(q_obj_b[1]) + float(q_obj_b[3])*float(q_obj_b[2]))
                    cz = 2*(float(q_obj_b[0])*float(q_obj_b[2]) - float(q_obj_b[3])*float(q_obj_b[1]))
                    

                    # if cx_list is not None and len(cx_list) > 1 and mean_cx is not None:
                        
                    #     if cx >= mean_cx + 0.05 or cx <= mean_cx - 0.05:
                    #         cx = cx_list[-1]

                    #     if cy >= mean_cy + 0.05 or cy <= mean_cy - 0.05:
                    #         cy = cy_list[-1]
                        
                    #     if cz >= mean_cz + 0.05 or cz <= mean_cz - 0.05:
                    #         cz = cz_list[-1]


                    cam_vec = np.array([[cx, cy, cz]]).T
                    cam_vec *= 1/(np.linalg.norm(cam_vec))
                                     

                    #Draw ArUco contourn and Axis
                    cv.aruco.drawAxis(image, self.mtx1, self.dist1, rvec_obj, tvec_obj, 0.185)
                    cv.aruco.drawDetectedMarkers(image, markerCorners[0])

        return position, q_obj_b, cam_vec, image
    
    #Camera Processing
    def img_show(self, step):

        rvecs = None
        tvecs = None

        rvecs2 = None
        tvecs2 = None

        x_Fold = 0
        y_Fold = 0
        z_Fold = 0
        
        global T_cr

        # T_cr = np.eye(4)

        #Font setup
        font = cv.FONT_HERSHEY_PLAIN
        
        #Load the predefinied dictionary
        dictionary = cv.aruco.Dictionary_get(cv.aruco.DICT_4X4_50)

        if step % self.cv_cam.frame_int == 0:           
            ret, image = self.cv_cam.get_image()
            ret, image2 = self.cv_cam_2.get_image()

            if ret:
        
        
        ####################################### ARUCO MARKER POSE ESTIMATION #########################################################################


                self.position, self.quat, self.cam_vec, image = self.aruco_detection(image, dictionary, rvecs, tvecs, self.cx_list, self.cy_list, self.cz_list, self.mean_cx, self.mean_cy, self.mean_cz)
                
                if self.cam_vec is not None:
                    self.cx_list.append(float(self.cam_vec[0]))
                    self.cy_list.append(float(self.cam_vec[1]))
                    self.cz_list.append(float(self.cam_vec[2]))

                    self.mean_cx = statistics.mean(self.cx_list)
                    self.mean_cy = statistics.mean(self.cy_list)
                    self.mean_cz = statistics.mean(self.cz_list)

                self.position2, self.quat2, self.cam_vec2, image2 = self.aruco_detection(image2, dictionary, rvecs2, tvecs2, self.cx_list2, self.cy_list2, self.cz_list2, self.mean2_cx, self.mean2_cy, self.mean2_cz)

                if self.cam_vec2 is not None:
                    self.cx_list2.append(float(self.cam_vec2[0]))
                    self.cy_list2.append(float(self.cam_vec2[1]))
                    self.cz_list2.append(float(self.cam_vec2[2]))

                    self.mean2_cx = statistics.mean(self.cx_list2)
                    self.mean2_cy = statistics.mean(self.cy_list2)
                    self.mean2_cz = statistics.mean(self.cz_list2)

                # self.cam_vec, self.cam_vec2 = None, None

                self.position = None
                # self.position2 = None

                if self.position is not None:
                    #Corrected Position Camera 1

                    self.position[0] = self.position[0]*0.968 + 0.0845
                    self.position[1] = self.position[1]*0.895 + 0.632
                    self.position[2] = self.position[2]*0.83 - 1.71

                    # Z correction
                    self.position[0] -= -self.position[2]*0.0299 + 0.0559
                    self.position[1] -= -self.position[2]*0.0112 + 0.0114

                    # print('Posição 1:', self.position.T)

                if self.position2 is not None:
                    #Corrected Position Camera 2
                    
                    self.position2[0] = self.position2[0]*0.946 + 0.00334
                    self.position2[1] = self.position2[1]*0.892 - 0.656
                    self.position2[2] = self.position2[2]*0.815 - 1.55

                    # Z correction
                    self.position2[0] -= -self.position2[2]*0.00821 + 0.00828
                    self.position2[1] -= self.position2[2]*0.025 - 0.0366

                    # print('Posição 2:', self.position2.T)
                
                if self.position is not None and self.position2 is not None:

                    self.position_mean = self.position*0.9 + self.position2*0.1

                if self.position is not None and self.position2 is None:

                    self.position_mean = self.position
                
                if self.position is None and self.position2 is not None:

                    self.position_mean = self.position2

                # if step % 10 == 0:

                # self.cx_list2.append(float(self.cam_vec[0]))
                # self.cy_list2.append(float(self.cam_vec[1]))
                # self.cz_list2.append(float(self.cam_vec[2]))

                # if len(self.cx_list2) == 2000:

                #     std_cx = np.std(self.cx_list2)
                #     std_cy = np.std(self.cy_list2)
                #     std_cz = np.std(self.cz_list2)

                #     print('Standard Deviation: \n cx_C2 = {0} \n cy_C2 = {1} \n cz_C2 = {2}'.format(std_cx, std_cy, std_cz))

                cv.imshow('Drone Camera',image)
                cv.imshow('Drone Camera 2', image2)
                cv.waitKey(1)

        