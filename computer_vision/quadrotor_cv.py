import cv2 as cv
import numpy as np
from numpy.linalg import inv, det
import scipy as sci
from scipy.spatial.transform import Rotation as R
from collections import deque
import math
import time
from matplotlib import pyplot as plt
from computer_vision.detector_setup import detection_setup
from computer_vision.img_2_cv import opencv_camera


class computer_vision():
    def __init__(self, render, quad_model, cv_cam, cv_cam_2, camera_cal1, camera_cal2, quad_position):
        
        # self.fast, self.criteria, self.nCornersCols, self.nCornersRows, self.objp, self.checker_scale, self.checker_sqr_size = detection_setup(render)

        self.mtx1 = camera_cal1.mtx
        self.dist1 = camera_cal1.dist
        print(self.dist1)

        self.render = render  
        self.quad_position = quad_position
        self.render.quad_model.setPos(0, 0, 0)
        self.render.quad_model.setHpr(0, 0, 0)
        
        self.cv_cam = cv_cam
        self.cv_cam.cam.node().getLens().setFilmSize(36, 24)
        self.cv_cam.cam.node().getLens().setFocalLength(40)
        self.cv_cam.cam.setPos(0, 0, 6.5)
        self.cv_cam.cam.lookAt(0, 0, 0)
        self.cv_cam.cam.reparentTo(self.render.render)


        self.obj_frame = []
        self.ground_frame = []
        self.distances = []
        self.T_flag = False

        
        # self.pos_x = 1
        # self.pos_y = 1
        # self.pos_z = 4.5

        self.render.taskMgr.add(self.img_show, 'OpenCv Image Show')
        # self.render.taskMgr.add(self.cam_displacement, 'Cam Displacement')

    def cam_displacement(self, task):
        while task.time < 20:
            self.pos_x -= 0.0
            self.pos_y += 0.0
            self.pos_z += 0.01
            self.cv_cam.cam.setPos(self.pos_x, self.pos_y, self.pos_z)
            return task.cont
        print('Done')
        return task.done
    
    def computeAngles(self, q0, q1, q2, q3):

        roll = 180*math.atan2(q0*q1 + q2*q3, 0.5 - q1*q1 - q2*q2)/math.pi
        pitch = 180*math.asin(-2.0 * (q1*q3 - q0*q2))/math.pi
        yaw = 180*math.atan2(q1*q2 + q0*q3, 0.5 - q2*q2 - q3*q3)/math.pi

        return roll, pitch, yaw

    def img_show(self, task):


        rvecs = None
        tvecs = None

        global T_cr

        # T_cr = np.eye(4)

        #Font setup
        font = cv.FONT_HERSHEY_PLAIN
        
        #Load the predefinied dictionary
        dictionary = cv.aruco.Dictionary_get(cv.aruco.DICT_4X4_50)

        if task.frame % self.cv_cam.frame_int == 1:           
            ret, image = self.cv_cam.get_image()
            #ret, image2 = self.cv_cam_2.get_image()

            ##Chessboard 
            if ret:
        
        
        ####################################### ARUCO MARKER POSE ESTIMATION #########################################################################

                image = cv.cvtColor(image, cv.COLOR_BGRA2BGR)
                gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

                #Initialize the detector parameters using defaults values
                parameters = cv.aruco.DetectorParameters_create()
                parameters.adaptiveThreshWinSizeMin = 3
                parameters.adaptiveThreshWinSizeMax = 15
                parameters.adaptiveThreshWinSizeStep = 3
                parameters.adaptiveThreshConstant = 28

                parameters.cornerRefinementWinSize = 5
                parameters.cornerRefinementMethod = cv.aruco.CORNER_REFINE_CONTOUR
                # parameters.cornerRefinementMaxIterations = 20
                parameters.cornerRefinementMinAccuracy = 0.1
                # parameters = self.detector_aruco_parameters(10, 23, 10, 7)

                #Detect the markers in the image
                markerCorners, markerIDs, rejectedCandidates = cv.aruco.detectMarkers(gray, dictionary, parameters=parameters)
                
                #Descending order Marker ID's array
                if len(markerIDs)==2:
                    order = np.argsort(-markerIDs.reshape(1,2))
                    markerIDs = markerIDs[order].reshape(2,1)
                    markerCorners = np.asarray(markerCorners, dtype='float32')[order]

                #If there is a marker compute the pose
                if markerIDs is not None and len(markerIDs)==2:
                    print("Marker Detected")

                    for i in range(0, len(markerIDs)):
                        print("---------")
                        print(i)

                        if markerIDs[i]==10:

                            # print("Marker Corners Reference:")
                            # print(markerCorners[0][i])

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

                            r_ref = sci.spatial.transform.Rotation.from_matrix(T_cr[0:3, 0:3])
                            q_ref = r_ref.as_quat()
                            roll_ref, pitch_ref, yaw_ref = self.computeAngles(q_ref[3], q_ref[0], q_ref[1], q_ref[2])
                            euler_ref = np.array([roll_ref, pitch_ref, yaw_ref])
                            print("Euler Reference:")
                            print(euler_ref)

                            cv.aruco.drawAxis(image, self.mtx1, self.dist1, rvec_ref, tvec_ref, 0.5)
                            cv.aruco.drawDetectedMarkers(image, markerCorners[0])
        
                        
                        if markerIDs[i]==4:

                            # print("Marker Corners Movement:")
                            # print(markerCorners[0][i])

                            rvec_obj, tvec_obj, _ = cv.aruco.estimatePoseSingleMarkers(markerCorners[0][i], 0.0925, self.mtx1, self.dist1, rvecs, tvecs)
                            rvec_obj = np.reshape(rvec_obj, (3,1))
                            tvec_obj = np.reshape(tvec_obj, (3,1))

                            #Use Rodrigues formula to transform rotation vector into matrix
                            R_dc, _ = cv.Rodrigues(rvec_obj)

                            #Homogeneous Transformation Object Frame to Camera Frame
                            last_col = np.array([[0, 0, 0, 1]])
                            T_dc = np.concatenate((R_dc, tvec_obj), axis=1)
                            T_dc = np.concatenate((T_dc, last_col), axis=0)

                            r_mov = sci.spatial.transform.Rotation.from_matrix(T_dc[0:3, 0:3])
                            q_mov = r_mov.as_quat()
                            roll_mov, pitch_mov, yaw_mov = self.computeAngles(q_mov[3], q_mov[0], q_mov[1], q_mov[2])
                            euler_mov = np.array([roll_mov, pitch_mov, yaw_mov])
                            print("Euler Movement")
                            print(euler_mov)

                            if T_cr is not None:
                                #Homogeneous Transformation Object Frame to Fixed Frame
                                T_dr = T_cr@T_dc
                            else:
                                T_dr = np.eye(4)@T_dc
      
                            #Getting quaternions from rotation matrix
                            r_obj = sci.spatial.transform.Rotation.from_matrix(T_dr[0:3, 0:3])
                            q_obj = r_obj.as_quat()
                            print(q_obj)
                            roll_obj, pitch_obj, yaw_obj = self.computeAngles(q_obj[3], q_obj[0], q_obj[1], q_obj[2])
                            euler_obj = np.array([roll_obj, pitch_obj, yaw_obj])

                            xf_obj = float(T_dr[0,3])
                            yf_obj = float(T_dr[1,3])
                            zf_obj = float(T_dr[2,3])


                            cv.aruco.drawAxis(image, self.mtx1, self.dist1, rvec_obj, tvec_obj, 0.0925)
                            # print(tvec_obj)
                            cv.aruco.drawDetectedMarkers(image, markerCorners[0])


                            modulo = np.sqrt(q_obj[0]**2 + q_obj[1]**2 + q_obj[2]**2 + q_obj[3]**2)
                            #Print position values in frame
                            cv.putText(image, "X:"+str(np.round(float(xf_obj), 4)), (10,500), font, 1, (255,255,255), 2)
                            cv.putText(image, "Y:"+str(np.round(float(yf_obj), 4)), (100,500), font, 1, (255,255,255), 2)
                            cv.putText(image, "Z:"+str(np.round(float(zf_obj), 4)), (190,500), font, 1, (255,255,255), 2)
                            
                            cv.putText(image, "Orientacao Estimada por Camera:", (10, 20), font, 1, (255, 255, 255), 2)
                            cv.putText(image, "Phi:"+str(np.round(float(euler_obj[0]), 2)), (10,40), font, 1, (0,0,255), 2)
                            cv.putText(image, "Theta:"+str(np.round(float(euler_obj[1]), 2)), (10,60), font, 1, (0,255,0), 2)
                            cv.putText(image, "Psi:"+str(np.round(float(euler_obj[2]), 2)), (10,80), font, 1, (255,0,0), 2)
                            
                            cv.putText(image, "q0:"+str(np.round(float(q_obj[3]), 3)), (10,400), font, 1, (255,255,255), 2)
                            cv.putText(image, "q1:"+str(np.round(float(q_obj[0]), 3)), (10,420), font, 1, (255,255,255), 2)
                            cv.putText(image, "q2:"+str(np.round(float(q_obj[1]), 3)), (10,440), font, 1, (255,255,255), 2)
                            cv.putText(image, "q3:"+str(np.round(float(q_obj[2]), 3)), (10,460), font, 1, (255,255,255), 2)
                            cv.putText(image, "Modulo:" + str(np.round(float(modulo), 3)), (10, 480), font, 1, (255, 0, 0),2)
                    
                    

                cv.imshow('Drone Camera',image)
                cv.waitKey(1)

        return task.cont