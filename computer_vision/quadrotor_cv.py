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
from computer_vision.quat_utils import Quat2Rot, QuatProd, computeAngles, SkewMat, DerivQuat



class computer_vision():
    def __init__(self, render, quad_model, cv_cam, cv_cam_2, camera_cal1, camera_cal2, quad_position):
        


        self.mtx1 = camera_cal1.mtx
        self.dist1 = camera_cal1.dist
        print(self.dist1)

        #Create attitude data.txt
        self.angle_data = open("angle_data.txt", 'w')
        self.kfquat_data = open("kfquat_data.txt", "w")
        self.kfeuler_data = open("kfeuler_data.txt", "w")
        self.pos_data = open('pos_data.txt', 'w')

        self.render = render  
        self.quad_position = quad_position
        self.render.quad_model.setPos(0, 0, 0)
        self.render.quad_model.setHpr(0, 0, 0)
        
        #Camera Setup
        self.cv_cam = cv_cam
        self.cv_cam.cam.node().getLens().setFilmSize(36, 24)
        self.cv_cam.cam.node().getLens().setFocalLength(40)
        self.cv_cam.cam.setPos(0, -2.1, 7.5)
        self.cv_cam.cam.setHpr(0, 300, 0)
        # self.cv_cam.cam.lookAt(0, 0, 0)
        self.cv_cam.cam.reparentTo(self.render.render)


        #Some local variables
        self.cx_list, self.cy_list, self.cz_list = [], [], []
        self.ax_list, self.ay_list, self.az_list = [], [], []
        self.index = 0
        self.index2 = 0
        self.time = 0
        self.camera_meas_flag = False
        

        #MEKF Constants
        self.var_a = 0.0035**2
        self.var_c = 0.0023**2
        self.var_v = 0.035**2
        self.var_u = 0.00015**2
        self.b_k = np.array([[0, 0, 0]], dtype='float32').T
        self.P_k = np.array([[0.1, 0, 0, 0, 0, 0],
                             [0, .1, 0, 0, 0, 0],
                             [0, 0, .1, 0, 0, 0],
                             [0, 0, 0, .01, 0, 0],
                             [0, 0, 0, 0, .01, 0],
                             [0, 0, 0, 0, 0, .01]])
        self.dx_k = np.array([[0, 0, 0, 0, 0, 0]], dtype='float32').T
        self.dt = self.quad_position.env.t_step

        #Run Tasks
        self.render.taskMgr.add(self.img_show, 'OpenCv Image Show')
        self.render.taskMgr.add(self.MEKF, 'MEKF')

    #Multiplicative Extended Kalman Filter
    def MEKF(self, task):
        
        #Get initial states when change episode's simulation
        if self.time % 6 == 0:
            print('oi')
            q_k0, R0 = self.quad_position.sensor.triad()
            self.q_k = np.array([[q_k0[1], q_k0[2], q_k0[3], q_k0[0]]], dtype='float32').T


        #.txt archives
        self.kfquat_data = open("kfquat_data.txt", "a+")
        self.kfeuler_data = open("kfeuler_data.txt", "a+")
        
        #Measurement Update, Quaternion Update and Bias Update
        q_K, b_K, P_K = self.Update_MEKF(self.dx_k, self.q_k, self.b_k, self.P_k)
        #Predict Quaternion using Gyro Measuments
        q_k, P_k = self.Prediction_MEKF(q_K, P_K, b_K)

        #Reset and propagation
        self.q_k = q_k
        self.b_k = b_K
        self.P_k = P_k
        self.dx_k = np.array([[0, 0, 0, 0, 0, 0]], dtype='float32').T
        self.camera_meas_flag = False


        #Get Real Attitude from Quadrotor
        attitude_real = self.quad_position.env.mat_rot
        r_real = sci.spatial.transform.Rotation.from_matrix(attitude_real)
        q_real = r_real.as_quat()
        roll_real, pitch_real, yaw_real = computeAngles(q_real[3], q_real[0], q_real[1], q_real[2])

        # print('---------------------------')
        # print('Real:', q_real.T)
        # print('Estimated:', self.q_k.T)

        #Get Euler Angles from Estimated MEKF Quaternion
        roll_est, pitch_est, yaw_est = computeAngles(self.q_k[3], self.q_k[0], self.q_k[1], self.q_k[2])

        # print(q_real)
        # print(self.q_k.T)

        if self.time <= 6:

            #Write Real and Estimated Quaternion Attitude in .txt archive
            self.kfquat_data.write("{:.4f} , ".format(float(q_K[3])) + "{:.4f} , ".format(float(q_K[0])) + "{:.4f} , ".format(float(q_K[1]))+ "{:.4f} , ".format(float(q_K[2])) + "{:.4f} , ".format(float(q_real[3])) + "{:.4f} , ".format(float(q_real[0])) + "{:.4f} , ".format(float(q_real[1])) + "{:.4f} , ".format(float(q_real[2])) + "{:.2f}".format(self.time) + "\n")

            #Write Real and Estimated Euler Attitude in .txt archive
            self.kfeuler_data.write("{:.4f} , ".format(float(roll_est)) + "{:.4f} , ".format(float(pitch_est)) + "{:.4f} , ".format(float(yaw_est))+ "{:.4f} , ".format(float(roll_real)) + "{:.4f} , ".format(float(pitch_real)) + "{:.4f} , ".format(float(yaw_real)) + "{:.2f}".format(self.time) + "\n")


        self.time += self.quad_position.env.t_step
        # print(self.time)
        self.kfquat_data.close()
        self.kfeuler_data.close()

        return task.cont

    def Prediction_MEKF(self, q_K, P_K, b_K):

        gyro = self.quad_position.sensor.gyro()

        # Discrete Quaternion Propagation
        omega_hat_k = gyro - b_K
        omega_hat_k_norm = np.linalg.norm(omega_hat_k)

        # dot_q = DerivQuat(omega_hat_k, q_K)
        # q_kp1 = q_K + dot_q*self.dt
        # recipNorm = 1/(np.linalg.norm(q_kp1))
        # q_kp1 *= recipNorm
        # print('Euler:', q_kp1)

        Psi_K = (omega_hat_k/omega_hat_k_norm)*np.sin(0.5*omega_hat_k_norm*self.dt)  
          
        Omega22 = np.array([[np.cos(0.5*omega_hat_k_norm*self.dt)]], dtype='float32')
        Omega21 = -Psi_K.T
        Omega12 = Psi_K
        Omega11 = np.cos(0.5*omega_hat_k_norm*self.dt)*np.eye(3) - SkewMat(Psi_K)
        Omega_up = np.concatenate((Omega11, Omega12), axis=1)
        Omega_down = np.concatenate((Omega21, Omega22), axis=1)
        Omega = np.concatenate((Omega_up, Omega_down), axis=0)
        q_kp1 =  Omega@q_K
        q_kp1 *= 1/(np.linalg.norm(q_kp1))

        #Discrete error-state transition matrix

        Phi11 = np.eye(3) - (np.sin(omega_hat_k_norm*self.dt)/omega_hat_k_norm)*SkewMat(omega_hat_k) + SkewMat(omega_hat_k)@SkewMat(omega_hat_k)*((1 - np.cos(omega_hat_k_norm*self.dt)/(omega_hat_k_norm**2)))

        Phi12 = ((1 - np.cos(omega_hat_k_norm*self.dt)/(omega_hat_k_norm**2)))*SkewMat(omega_hat_k) - np.eye(3)*self.dt - ((omega_hat_k_norm*self.dt - np.sin(omega_hat_k_norm*self.dt))/(omega_hat_k_norm**3))*SkewMat(omega_hat_k)@SkewMat(omega_hat_k)

        Phi21 = np.zeros((3,3))
        Phi22 = np.eye(3)

        Phi_up = np.concatenate((Phi11, Phi12), axis=1) 
        Phi_down = np.concatenate((Phi21, Phi22), axis=1)
        Phi = np.concatenate((Phi_up, Phi_down), axis=0)
        
        #Discrete input matrix

        Gamma = np.array([[-1, 0, 0, 0, 0, 0],
                          [0, -1, 0, 0, 0, 0],
                          [0, 0, -1, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 1]])
        
        #Discrete noise process covariance matrix

        Q11 = (self.var_v*self.dt + (1/3)*self.var_u*self.dt**3)*np.eye(3)
        Q12 = 0.5*self.var_u*self.dt**2*np.eye(3)
        Q21 = Q12
        Q22 = self.var_u*self.dt*np.eye(3)
        Q_up = np.concatenate((Q11, Q12), axis=1)
        Q_down = np.concatenate((Q21, Q22), axis=1)
        Q = np.concatenate((Q_up, Q_down), axis=0)

        #Discrete Covariance Propagation
        P_kp1 = Phi@P_K@Phi.T + Gamma@Q@Gamma.T

        return q_kp1, P_kp1

    def Update_MEKF(self, dx_k, q_k, b_k, P_k):
        
        #Estimated Rotation Matrix - Global to Local - Inertial to Body
        A_q = Quat2Rot(q_k)


        if self.camera_meas_flag:
            #Current Sensor Measurement

            #Accelerometer
            induced_acceleration = self.quad_position.env.f_in.flatten()/1.03 - (A_q @ np.array([[0, 0, -9.82]]).T).flatten()
            accel_grav = self.quad_position.sensor.accel() - induced_acceleration
            b_a = np.array([accel_grav], dtype='float32').T
            b_a *= 1/(np.linalg.norm(b_a))

            #Camera
            b_c = self.cam_vec
            #Measurement Sensor Vector
            b = np.concatenate((b_a, b_c), axis=0)

            #Reference Accelerometer Direction (Gravitational Acceleration)
            r_a = np.array([[0, 0, -1]]).T
            #Reference Camera Direction
            r_c = np.array([[1, 0, 0]]).T
            #Camera Estimated Output
            b_hat_c = A_q @ r_c
            #Accelerometer Estimated Output
            b_hat_a = A_q @ r_a
            #Estimated Measurement Vector
            b_hat = np.concatenate((b_hat_a, b_hat_c), axis=0)

            #Sensitivy Matrix Accelerometer
            H_ka_left = SkewMat(b_hat_a)
            H_ka_right = np.zeros((3,3))
            H_ka = np.concatenate((H_ka_left, H_ka_right), axis=1)
            
            #Sensitivy Matrix Camera   
            H_kc_left = SkewMat(b_hat_c)
            H_kc_right = np.zeros((3,3))
            H_kc = np.concatenate((H_kc_left, H_kc_right), axis=1)

            #General Sensitivy Matrix
            H_k = np.concatenate((H_ka, H_kc), axis=0)

            #Accelerometer Measurement Covariance Matrix
            Ra = np.array([[self.var_a, 0, 0],
                            [0, self.var_a, 0],
                            [0, 0, self.var_a]], dtype='float32')
            
            
            #Camera Measurement Covariance Matrix
            Rc = np.array([[self.var_c, 0, 0],
                            [0, self.var_c, 0],
                            [0, 0, self.var_c]], dtype='float32')
            
            #General Measurement Covariance Matrix
            R_top = np.concatenate((Ra, np.zeros((3,3))), axis=1)
            R_down = np.concatenate((np.zeros((3,3)), Rc), axis=1)
            R = np.concatenate((R_top, R_down), axis=0) 

            #Kalman Gain
            K_k = P_k@H_k.T @ inv(H_k@P_k@H_k.T + R)

            #Update Covariance
            P_K = (np.eye(6) - K_k@H_k)@P_k

            #Inovation (Residual)
            e_k = b - b_hat

            #Update State
            dx_K = K_k@e_k
        
        else:
            #Current Sensor Measurement

            #Accelerometer 
            induced_acceleration = self.quad_position.env.f_in.flatten()/1.03 - (A_q @ np.array([[0, 0, -9.82]]).T).flatten()
            accel_grav = self.quad_position.sensor.accel() - induced_acceleration
            b_a = np.array([accel_grav], dtype='float32').T
            b_a *= 1/(np.linalg.norm(b_a))

            #Reference Accelerometer Direction (Gravitational Acceleration)
            r_a = np.array([[0, 0, -1]]).T
            #Accelerometer Estimated Output
            b_hat_a = A_q @ r_a

            #Sensitivy Matrix Accelerometer
            H_ka_left = SkewMat(b_hat_a)
            H_ka_right = np.zeros((3,3))
            H_ka = np.concatenate((H_ka_left, H_ka_right), axis=1)

            #Accelerometer Measurement Covariance Matrix
            Ra = np.array([[self.var_a, 0, 0],
                            [0, self.var_a, 0],
                            [0, 0, self.var_a]], dtype='float32')

            #Kalman Gain
            K_k = P_k@H_ka.T @ inv(H_ka@P_k@H_ka.T + Ra)

            #Update Covariance
            P_K = (np.eye(6) - K_k@H_ka)@P_k

            #Inovation (Residual)
            e_k = b_a - b_hat_a
  
            #Update State
            dx_K = K_k@e_k
        
        #Update Quaternion
        dq_k = dx_K[0:3,:]
        q_K = q_k + DerivQuat(dq_k, q_k)
        q_K *= 1/(np.linalg.norm(q_K))    

        #Update Biases
        db = dx_K[3:6,:]
        b_K = b_k + db   

        # print(q_K)

        return q_K, b_K, P_K

    
    #Camera Processing
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

            if ret:
        
        
        ####################################### ARUCO MARKER POSE ESTIMATION #########################################################################

                image = cv.cvtColor(image, cv.COLOR_BGRA2BGR)
                gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

                #Initialize the detector parameters using defaults values
                parameters = cv.aruco.DetectorParameters_create()
                parameters.adaptiveThreshWinSizeMin = 3
                parameters.adaptiveThreshWinSizeMax = 15
                parameters.adaptiveThreshWinSizeStep = 3
                parameters.adaptiveThreshConstant = 20

                parameters.cornerRefinementWinSize = 10
                parameters.cornerRefinementMethod = cv.aruco.CORNER_REFINE_CONTOUR
                # parameters.cornerRefinementMaxIterations = 20
                parameters.cornerRefinementMinAccuracy = 0.0


                #.txt archives
                self.angle_data = open("angle_data.txt", "a+")
                self.pos_data = open('pos_data.txt', 'a+')

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
                            roll_ref, pitch_ref, yaw_ref = computeAngles(q_ref[3], q_ref[0], q_ref[1], q_ref[2])
                            euler_ref = np.array([roll_ref, pitch_ref, yaw_ref])

                            #Draw axis in marker
                            cv.aruco.drawAxis(image, self.mtx1, self.dist1, rvec_ref, tvec_ref, 0.5)
                            cv.aruco.drawDetectedMarkers(image, markerCorners[0])
        
                        #Check if it's moving marker/object marker
                        if markerIDs[i]==4:

                            #Get Pose Estimation of the Moving Marker
                            rvec_obj, tvec_obj, _ = cv.aruco.estimatePoseSingleMarkers(markerCorners[0][i], 0.185, self.mtx1, self.dist1, rvecs, tvecs)
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
                            roll_mov, pitch_mov, yaw_mov = computeAngles(q_mov[3], q_mov[0], q_mov[1], q_mov[2])
                            euler_mov = np.array([roll_mov, pitch_mov, yaw_mov])

                            if T_cr is not None:
                                #Homogeneous Transformation Object Frame to Fixed Frame
                                T_dr = T_cr@T_dc
                            else:
                                T_dr = np.eye(4)@T_dc

                            T_rd = T_dr.T

                            #Getting quaternions from rotation matrix
                            r_obj = sci.spatial.transform.Rotation.from_matrix(T_dr[0:3, 0:3])
                            q_obj = r_obj.as_quat()
                            # print(q_obj)
                            roll_obj, pitch_obj, yaw_obj = computeAngles(q_obj[3], q_obj[0], q_obj[1], q_obj[2])
                            euler_obj = np.array([roll_obj, pitch_obj, yaw_obj])
                            
                            r_obj_b = sci.spatial.transform.Rotation.from_matrix(T_rd[0:3, 0:3])
                            q_obj_b = r_obj_b.as_quat()

                            #Real Quadcopter Attitude
                            attitude_real = self.quad_position.env.mat_rot
                            r_real = sci.spatial.transform.Rotation.from_matrix(attitude_real)
                            q_real = r_real.as_quat()
                            roll_real, pitch_real, yaw_real = computeAngles(q_real[3], q_real[0], q_real[1], q_real[2])

                            # print(T_dr)

                            #Marker's Position
                            zf_obj = float(T_dr[2,3])*0.868 - 2.52
                            xf_obj = float(T_dr[0,3])*0.978 - 0.0563
                            yf_obj = float(T_dr[1,3])*0.932 + 0.725

                            #Measurement Direction Vector
                            
                            # cx = 2*(q_obj_b[0]*q_obj_b[1] - q_obj_b[3]*q_obj_b[2])
                            # cy = q_obj_b[3]**2 - q_obj_b[0]**2 + q_obj_b[1]**2 - q_obj_b[2]**2
                            # cz = 2*(q_obj_b[1]*q_obj_b[2] + q_obj_b[3]*q_obj_b[0])
                            cx = q_obj_b[3]**2 + q_obj_b[0]**2 - q_obj_b[1]**2 - q_obj_b[2]**2
                            cy = 2*(q_obj_b[0]*q_obj_b[1] + q_obj_b[3]*q_obj_b[2])
                            cz = 2*(q_obj_b[0]*q_obj_b[2] - q_obj_b[3]*q_obj_b[1])
                            self.cam_vec = np.array([[cx, cy, cz]]).T
                            self.cam_vec *= 1/(np.linalg.norm(self.cam_vec))

                            # self.camera_meas_flag = True

                            #Get the standard deviation from camera and accelerometer

                            # induced_acceleration = self.quad_position.env.f_in.flatten()/1.03 - (self.quad_position.sensor.triad()[1] @ np.array([[0, 0, -9.82]]).T).flatten()
                            # gravity_body = self.quad_position.sensor.accel() - induced_acceleration
                            # accel = np.array([gravity_body],dtype='float32').T
                            # accel *= 1/(np.linalg.norm(accel))
                            # accel_x = accel[0]
                            # accel_y = accel[1]
                            # accel_z = accel[2]

                            # self.cx_list.append(cx)
                            # self.cy_list.append(cy)
                            # self.cz_list.append(cz)
                            # self.ax_list.append(float(accel_x))
                            # self.ay_list.append(float(accel_y))
                            # self.az_list.append(float(accel_z))


                            # if len(self.cx_list) == 5000 and len(self.ax_list) == 5000:

                            #     cx_std = statistics.stdev(self.cx_list)
                            #     cy_std = statistics.stdev(self.cy_list)
                            #     cz_std = statistics.stdev(self.cz_list)
                            #     ax_std = statistics.stdev(self.ax_list)
                            #     ay_std = statistics.stdev(self.ay_list)
                            #     az_std = statistics.stdev(self.az_list)

                            #     print('cx SD: {0} \n cy SD: {1} \n cz SD: {2}'.format(cx_std, cy_std, cz_std))
                            #     print('ax SD: {0} \n ay SD: {1} \n az SD: {2}'.format(ax_std, ay_std, az_std))

                            #     self.cx_list, self.cy_list, self.cz_list = [], [], []
                            #     self.ax_list, self.ay_list, self.az_list = [], [], []

                            #Real Quadcopter's Position
                            x_real = self.quad_position.env.state[0]
                            y_real = self.quad_position.env.state[2]
                            z_real = self.quad_position.env.state[4] + 5
                            
                            #Position Error
                            error_x = x_real - xf_obj
                            error_y = y_real - yf_obj
                            error_z = z_real - zf_obj

                            #Draw ArUco contourn and Axis
                            cv.aruco.drawAxis(image, self.mtx1, self.dist1, rvec_obj, tvec_obj, 0.185)
                            cv.aruco.drawDetectedMarkers(image, markerCorners[0])

                            # #Print position values in frame
                            # cv.putText(image, "X:"+str(np.round(float(xf_obj), 4)), (10,500), font, 1, (255,255,0), 2)
                            # cv.putText(image, "Y:"+str(np.round(float(yf_obj), 4)), (100,500), font, 1, (255,255,0), 2)
                            # cv.putText(image, "Z:"+str(np.round(float(zf_obj), 4)), (190,500), font, 1, (255,255,0), 2)
                            
                            # cv.putText(image, "Orientacao Estimada por Camera:", (10, 20), font, 1, (255, 255, 255), 2)
                            # cv.putText(image, "Phi:"+str(np.round(float(euler_obj[0]), 2)), (10,40), font, 1, (0,0,255), 2)
                            # cv.putText(image, "Theta:"+str(np.round(float(euler_obj[1]), 2)), (10,60), font, 1, (0,255,0), 2)
                            # cv.putText(image, "Psi:"+str(np.round(float(euler_obj[2]), 2)), (10,80), font, 1, (0,255,0), 2)

                            # cv.putText(image, "Orientacao Real:", (10, 120), font, 1, (255, 255, 255), 2)
                            # cv.putText(image, "Phi:"+str(np.round(float(roll_real), 2)), (10,140), font, 1, (0,0,255), 2)
                            # cv.putText(image, "Theta:"+str(np.round(float(pitch_real), 2)), (10,160), font, 1, (0,255,0), 2)
                            # cv.putText(image, "Psi:"+str(np.round(float(yaw_real), 2)), (10,180), font, 1, (255,0,0), 2)
                            
                            if self.time <= 6:
                                #Write Real and Estimated Attitude in .txt archive
                                self.angle_data.write("{:.2f} , ".format(float(euler_obj[0])) + "{:.2f} , ".format(float(euler_obj[1])) + "{:.2f} , ".format(float(euler_obj[2]))
                                    + "{:.2f} , ".format(float(roll_real)) + "{:.2f} , ".format(float(pitch_real)) + "{:.2f} , ".format(float(yaw_real)) + "{:.2f}".format(self.time) + "\n")
                                #Write Real and Estimated Position in .txt archive
                                self.pos_data.write("{:.4f} , ".format(float(xf_obj)) + "{:.4f} , ".format(float(yf_obj)) + "{:.4f} , ".format(float(zf_obj))
                                    + "{:.4f} , ".format(float(x_real)) + "{:.4f} , ".format(float(y_real)) + "{:.4f} , ".format(float(z_real)) + "{:.2f}".format(self.time) + "\n")

                
                cv.imshow('Drone Camera',image)
                cv.waitKey(1)
                self.angle_data.close()
                self.pos_data.close()

        return task.cont