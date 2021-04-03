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
from environment.quadrotor_env import sensor



class computer_vision():
    def __init__(self, render, quad_model, cv_cam, cv_cam_2, camera_cal1, camera_cal2, quad_position):
        
        # self.fast, self.criteria, self.nCornersCols, self.nCornersRows, self.objp, self.checker_scale, self.checker_sqr_size = detection_setup(render)

        self.mtx1 = camera_cal1.mtx
        self.dist1 = camera_cal1.dist
        print(self.dist1)

        #Create attitude data.txt
        self.angle_data = open("angle_data.txt", 'w')
        self.kfquat_data = open("kfquat_data.txt", "w")
        self.kfeuler_data = open("kfeuler_data.txt", "w")
        self.pos_data = open('pos_data.txt', 'w')
        self.angle_data.close()

        self.render = render  
        self.quad_position = quad_position
        self.render.quad_model.setPos(0, 0, 0)
        self.render.quad_model.setHpr(0, 0, 0)
        
        self.cv_cam = cv_cam
        self.cv_cam.cam.node().getLens().setFilmSize(36, 24)
        self.cv_cam.cam.node().getLens().setFocalLength(40)
        self.cv_cam.cam.setPos(0, -2.1, 7.5)
        self.cv_cam.cam.setHpr(0, 300, 0)
        # self.cv_cam.cam.lookAt(0, 0, 0)
        self.cv_cam.cam.reparentTo(self.render.render)


        self.x_vals = []
        self.y_vals = []
        self.index = 0
        self.index2 = 0
        self.distances = []
        self.T_flag = False

        self.q_ant = np.array([1, 0, 0, 0])
        

        #MEKF
        self.var_a = self.quad_position.sensor.a_std**2
        self.var_m = 0.1
        self.var_q = .01
        self.var_b = .01
        self.q_k = np.array([[1, 0, 0, 0]], dtype='float32').T
        self.b_k = np.array([[0, 0, 0]], dtype='float32').T
        self.P_k = np.eye(6)*0.25
        self.dx_k = np.array([[0, 0, 0, 0, 0, 0]], dtype='float32').T
        self.dt = 0.1
        self.cam_vec = np.array([[0, 1, 0]]).T


        self.render.taskMgr.add(self.img_show, 'OpenCv Image Show')
        self.render.taskMgr.add(self.MEKF, 'MEKF')
        # self.render.taskMgr.add(self.Madgwick_Update, 'Madgwick Filter')

    
    def QuatProd(self, p, q):
        
        pv = np.array([p[1], p[2], p[3]], dtype='float32')
        ps = p[0]

        qv = np.array([q[1], q[2], q[3]], dtype='float32')
        qs = q[0]

        scalar = ps*qs - pv.T@qv
        vector = ps*qv + qs*pv + np.cross(pv, qv, axis=0)

        q_res = np.concatenate((scalar, vector), axis=0)

        return q_res

    def computeAngles(self, q0, q1, q2, q3):

        roll = 180*math.atan2(q0*q1 + q2*q3, 0.5 - q1*q1 - q2*q2)/math.pi
        pitch = 180*math.asin(-2.0 * (q1*q3 - q0*q2))/math.pi
        yaw = 180*math.atan2(q1*q2 + q0*q3, 0.5 - q2*q2 - q3*q3)/math.pi

        return roll, pitch, yaw

    def MEKF(self, task):
        
        self.kfquat_data = open("kfquat_data.txt", "a+")
        self.kfeuler_data = open("kfeuler_data.txt", "a+")
        
        q_K, b_K, P_K = self.Update_MEKF(self.dx_k, self.q_k, self.b_k, self.P_k)
        q_k, P_k = self.Prediction_MEKF(q_K, P_K)

        self.q_k = q_k
        self.b_k = b_K
        self.P_k = P_k
        self.dx_k = np.array([[0, 0, 0, 0, 0, 0]], dtype='float32').T

        attitude_real = self.quad_position.env.mat_rot
        r_real = sci.spatial.transform.Rotation.from_matrix(attitude_real)
        q_real = r_real.as_quat()
        roll_real, pitch_real, yaw_real = self.computeAngles(q_real[3], q_real[0], q_real[1], q_real[2])

        roll_est, pitch_est, yaw_est = self.computeAngles(q_k[0], q_k[1], q_k[2], q_k[3])

        self.kfquat_data.write("{:.4f} , ".format(float(q_K[0])) + "{:.4f} , ".format(float(q_K[1])) + "{:.4f} , ".format(float(q_K[2]))+ "{:.4f} , ".format(float(q_K[3])) + "{:.4f} , ".format(float(q_real[3])) + "{:.4f} , ".format(float(q_real[0])) + "{:.4f} , ".format(float(q_real[1])) + "{:.4f} , ".format(float(q_real[2])) + "{:.2f}".format(self.index2) + "\n")

        self.kfeuler_data.write("{:.4f} , ".format(float(roll_est)) + "{:.4f} , ".format(float(pitch_est)) + "{:.4f} , ".format(float(yaw_est))+ "{:.4f} , ".format(float(roll_real)) + "{:.4f} , ".format(float(pitch_real)) + "{:.4f} , ".format(float(yaw_real)) + "{:.2f}".format(self.index2) + "\n")

        # print("Outer:", self.index2)
        self.index2 += 1
        self.kfquat_data.close()
        self.kfeuler_data.close()

        return task.cont

    def Prediction_MEKF(self, q_K, P_K):

        gyro = self.quad_position.sensor.gyro().reshape(3,1)
        #Quaternion Integration
        omega_hat_k = gyro - self.b_k

        dot_q = self.DerivQuat(omega_hat_k, q_K)

        q_k = q_K + dot_q*self.dt
        recipNorm = 1/(np.linalg.norm(q_k))
        q_k *= recipNorm
        
        #Covariance Propagation
        Phi = np.array([[0, omega_hat_k[2], -omega_hat_k[1], -1, 0, 0],
                        [-omega_hat_k[2], 0, omega_hat_k[0], 0, -1, 0],
                        [omega_hat_k[1], -omega_hat_k[0], 0, 0, 0, -1],
                        [0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 1]], dtype='float32')
        
        Gamma = np.array([[-1, 0, 0, 0, 0, 0],
                          [0, -1, 0, 0, 0, 0],
                          [0, 0, -1, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 1]])
        
        Q = np.array([[self.var_q*self.dt + (1/3)*self.var_b*self.dt**3, 0, 0, 0.5*self.var_b*self.dt**2, 0, 0],
                      [0, self.var_q*self.dt + (1/3)*self.var_b*self.dt**3, 0, 0, 0.5*self.var_b*self.dt**2, 0],
                      [0, 0, self.var_q*self.dt + (1/3)*self.var_b*self.dt**3, 0, 0, 0.5*self.var_b*self.dt**2],
                      [0.5*self.var_b*self.dt**2, 0, 0, self.var_b*self.dt, 0, 0],
                      [0, 0.5*self.var_b*self.dt**2, 0, 0, self.var_b*self.dt, 0],
                      [0, 0, 0.5*self.var_b*self.dt**2, 0, 0, self.var_b*self.dt]], dtype='float32')

        # Q = np.array([[self.var_q, 0, 0, 0, 0, 0],
        #               [0, self.var_q, 0, 0, 0, 0],
        #               [0, 0, self.var_q, 0, 0, 0],
        #               [0, 0, 0, self.var_b, 0, 0],
        #               [0, 0, 0, 0, self.var_b, 0],
        #               [0, 0, 0, 0, 0, self.var_b]], dtype='float32')

        P_k = Phi@P_K@Phi.T + Gamma@Q@Gamma.T

        return q_k, P_k

    def Update_MEKF(self, dx_k, q_k, b_k, P_k):
        
        #Propagate

        A_q = self.Quat2Rot(q_k)

        # print('Rot Matrix Quaternion Estimated: ', A_q)
        # print('Rot Matrix TRIAD: ', self.quad_position.sensor.triad()[1])
        # print('Rot Matrix Quadrotor: ', self.quad_position.env.mat_rot)

        #Murriel's Version (Read each sensor at a time)
        for i in range(0,2):
            
            #Accelerometer Measurement
            if i==0:
                #Measurement Update

                #Current Sensor Measurement
                b_a = self.quad_position.sensor.accel_grav().T

                #Reference Accelerometer Direction (Gravitational Acceleration)
                r_a = np.array([[0, 0, -1]]).T

                #Estimated Output
                b_hat_a = A_q @ r_a
     
                
                #Sensitivy Matrix       
                H_k = np.array([[0, -b_hat_a[2], b_hat_a[1], 0, 0, 0],
                                [b_hat_a[2], 0, -b_hat_a[0], 0, 0, 0],
                                [-b_hat_a[1], b_hat_a[0], 0, 0, 0, 0]], dtype='float32')
                
                #Measurement Covariance Matrix
                R = np.array([[self.var_a, 0, 0],
                              [0, self.var_a, 0],
                              [0, 0, self.var_a]], dtype='float32')
                
                #Kalman Gain
                K_k = P_k@H_k.T @ inv(H_k@P_k@H_k.T + R)

                #Update Covariance
                L = np.eye(6) - K_k@H_k
                P_K = L@P_k@L.T + K_k@R@K_k.T

                #Inovation (Residual)
                e_k = b_a - b_hat_a
                #Update State
                dx_K = dx_k + K_k@(e_k - H_k@dx_k)
                
                #Propagate values to Camera Measurement
                dx_k = dx_K
                P_k = P_K

            #Camera Measurement
            if i==1:
                
                #Measurement Update

                #Current Measurement
                b_c = self.cam_vec

                print('Camera Measurement:', b_c.T)

                #Reference direction
                # r_c = np.array([[0, math.sqrt(float(self.cam_vec[0])**2 + float(self.cam_vec[1])**2), float(self.cam_vec[2])]]).T
                r_c = np.array([[0, 1, 0]]).T

                b_hat_c = A_q @ r_c
                
                print('Estimated Measurement:', b_hat_c.T)

                #Sensitivy Matrix       
                H_k = np.array([[0, -b_hat_c[2], b_hat_c[1], 0, 0, 0],
                                [b_hat_c[2], 0, -b_hat_c[0], 0, 0, 0],
                                [-b_hat_c[1], b_hat_c[0], 0, 0, 0, 0]], dtype='float32')
                
                #Measurement Covariance Matrix
                R = np.array([[self.var_m, 0, 0],
                              [0, self.var_m, 0],
                              [0, 0, self.var_m]], dtype='float32')
        
                #Kalman Gain
                K_k = P_k@H_k.T @ inv(H_k@P_k@H_k.T + R)

                #Update Covariance
                L = np.eye(6) - K_k@H_k
                P_K = L@P_k@L.T + K_k@R@K_k.T

                #Inovation (Residual)
                e_k = b_c - b_hat_c
                print('Inovação:', e_k)
                #Update State
                dx_K = dx_k + K_k@(e_k - H_k@dx_k)

        #Update Quaternion
        dq_k = dx_K[0:3,:]
        q_K = q_k + self.DerivQuat(dq_k, q_k)
        q_K = q_K/(np.linalg.norm(q_K))    

        #Update Biases
        db = dx_K[3:6,:]
        b_K = b_k + db   

        # print(q_K)

        return q_K, b_K, P_K

    def Madgwick_Update(self, task):
        
        self.kf_data = open("kf_data.txt", "a+")

        g_meas = self.quad_position.sensor.gyro().reshape(3,1)
        a_meas = self.quad_position.sensor.accel_grav().reshape(3,1)
        m_meas = self.quad_position.sensor.mag_gauss().reshape(3,1)

        q = self.Madgwick(g_meas, a_meas, m_meas, self.dt, self.q_ant)

        self.q_ant = q

        attitude_real = self.quad_position.env.mat_rot
        r_real = sci.spatial.transform.Rotation.from_matrix(attitude_real)
        q_real = r_real.as_quat()
        roll_real, pitch_real, yaw_real = self.computeAngles(q_real[3], q_real[0], q_real[1], q_real[2])

        roll_est, pitch_est, yaw_est = self.computeAngles(q[0], q[1], q[2], q[3])

        # self.kf_data.write("{:.4f} , ".format(float(q_K[0])) + "{:.4f} , ".format(float(q_K[1])) + "{:.4f} , ".format(float(q_K[2]))+ "{:.4f} , ".format(float(q_K[3])) + "{:.4f} , ".format(float(q_real[3])) + "{:.4f} , ".format(float(q_real[0])) + "{:.4f} , ".format(float(q_real[1])) + "{:.4f} , ".format(float(q_real[2])) + "{:.2f}".format(self.index2) + "\n")

        self.kf_data.write("{:.4f} , ".format(float(roll_est)) + "{:.4f} , ".format(float(pitch_est)) + "{:.4f} , ".format(float(yaw_est))+ "{:.4f} , ".format(float(roll_real)) + "{:.4f} , ".format(float(pitch_real)) + "{:.4f} , ".format(float(yaw_real)) + "{:.2f}".format(self.index2) + "\n")

        self.quad_position.sensor.triad()

        print(self.index2)
        print("Erro X: {0}, Erro Y: {1}, Erro Z: {2}".format(roll_real-roll_est, pitch_real-pitch_est, yaw_real-yaw_est))
        self.index2 += 1
        self.kf_data.close()

        return task.cont

    def Madgwick(self, g, a, m, dt, q_ant):

        q0 = q_ant[0]
        q1 = q_ant[1]
        q2 = q_ant[2]
        q3 = q_ant[3]
        beta = 0.1

        #Gyroscope
        gx = g[0]
        gy = g[1]
        gz = g[2]
        #Accelerometer
        ax = a[0]
        ay = a[1]
        az = a[2]
        #Magnetometer
        mx = m[0]
        my = m[1]
        mz = m[2]

        # Rate of change of quaternion from gyroscope
        qDot1 = 0.5 * (-q1 * gx - q2 * gy - q3 * gz)
        qDot2 = 0.5 * (q0 * gx + q2 * gz - q3 * gy)
        qDot3 = 0.5 * (q0 * gy - q1 * gz + q3 * gx)
        qDot4 = 0.5 * (q0 * gz + q1 * gy - q2 * gx)

        # Compute feedback only if accelerometer measurement valid (avoids NaN in accelerometer normalisation)
        if ax !=0 and ay != 0 and az != 0:

            # # Normalise accelerometer measurement
            # recipNorm = 1/(math.sqrt(ax**2 + ay**2 + az**2))
            # ax *= recipNorm
            # ay *= recipNorm
            # az *= recipNorm

            # recipNorm = 1/(math.sqrt(mx**2 + my**2 + mz**2))
            # mx *= recipNorm
            # my *= recipNorm
            # mz *= recipNorm

            

            # Auxiliary variables to avoid repeated arithmetic
            _2q0 = 2.0 * q0
            _2q1 = 2.0 * q1
            _2q2 = 2.0 * q2
            _2q3 = 2.0 * q3
            _4q0 = 4.0 * q0
            _4q1 = 4.0 * q1
            _4q2 = 4.0 * q2
            _4q3 = 4.0 * q3
            _8q1 = 8.0 * q1
            _8q2 = 8.0 * q2
            q0q0 = q0 * q0
            q1q1 = q1 * q1
            q2q2 = q2 * q2
            q3q3 = q3 * q3

            # Reference direction of Earth's magnetic field
            hx = mx*(2*q0q0 - 1 + 2*q1q1) + my*(2*(q1*q2 + q0*q3)) + mz*(2*(q1*q3 - q0*q2))
            hy = mx*(2*(q1*q2 - q0*q3)) + my*(2*q0q0 - 1 + 2*q2q2) + mz*(2*(q2*q3 + q0*q1))
            hz = mx*(2*(q1*q3 + q0*q2)) + my*(2*(q2*q3 - q0*q1)) + mz*(2*q0q0 - 1 + 2*q3q3)
            bx = math.sqrt(hx**2 + hy**2)
            # bx = 1
            # bz = 0
            _2bx = 2*bx
            _2bz = 2*hz
            _4bx = 2.0 * _2bx
            _4bz = 2.0 * _2bz

            #Accelerometer 
            f_g = np.array([[-2*(q1*q3 - q0*q2) - ax],
                            [-2*(q0*q1 + q2*q3) - ay],
                            [-2*(0.5 - q1q1 - q2q2) - az]], dtype='float32').reshape(3,1)
            # print(f_g)
            J_g = np.array([[_2q2, -_2q3, _2q0, -_2q1],
                            [-_2q1, -_2q0, -_2q3, -_2q2],
                            [0, _4q1, _4q2, 0]], dtype='float32').reshape(3,4)

            #Magnetometer
            f_m = np.array([[_2bx*(0.5 - q2q2 - q3q3) + _2bz*(q1*q3 - q0*q2) - mx],
                            [_2bx*(q1*q2 - q0*q3) + _2bz*(q0*q1 + q2*q3) - my],
                            [_2bx*(q0*q2 + q1*q3) + _2bz*(0.5 - q1q1 - q2q2) - mz]], dtype='float32').reshape(3,1)
            print(f_m)
            J_m = np.array([[-_2bz*q2, _2bz*q3, -_4bx*q2 - _2bz*q0, -_4bx*q3 + _2bz*q1],
                            [-_2bx*q3 + _2bz*q1, _2bx*q2 + _2bz*q0, _2bx*q1 + _2bz*q3, -_2bx*q0 + _2bz*q2],
                            [_2bx*q2, _2bx*q3 - _4bz*q1, _2bx*q0 - _4bz*q2, _2bx*q1]], dtype='float32').reshape(3,4)

            # Gradient decent algorithm corrective step
            J_mg = np.concatenate((J_g, J_m), axis=0)
            f_mg = np.concatenate((f_g, f_m), axis=0).reshape(6,1)
            s = J_mg.T@f_mg
    
            s0 = s[0]
            s1 = s[1]
            s2 = s[2]
            s3 = s[3]
            recipNorm = 1/ (math.sqrt(s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3)) ## normalise step magnitude
            s0 *= recipNorm
            s1 *= recipNorm
            s2 *= recipNorm
            s3 *= recipNorm

            # Apply feedback step
            qDot1 -= beta * s0
            qDot2 -= beta * s1
            qDot3 -= beta * s2
            qDot4 -= beta * s3


        # Integrate rate of change of quaternion to yield quaternion
        q0 += qDot1 * dt
        q1 += qDot2 * dt
        q2 += qDot3 * dt
        q3 += qDot4 * dt

        # Normalise quaternion
        recipNorm = 1 / (math.sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3))
        q0 *= recipNorm
        q1 *= recipNorm
        q2 *= recipNorm
        q3 *= recipNorm

        q = np.array([q0, q1, q2, q3])

        # print('q0: {0}, q1: {1}, q2: {2}, q3: {3}'.format(q[0], q[1], q[2], q[3]))

        return q

    def Quat2Rot(self,q):

        qv = np.array([q[1], q[2], q[3]], dtype='float32')
        qs = q[0]
        
        qvx = np.array([[0, -qv[2], qv[1]],
                        [qv[2], 0, -qv[0]], 
                        [-qv[1], qv[0], 0]], dtype='float32') 

        A_q = (qs**2 - qv.T@qv)*np.eye(3) + 2*qv@qv.T + 2*qs*qvx

        return A_q

    def DerivQuat(self, w, q):

        wx = w[0]
        wy = w[1]
        wz = w[2]   
        omega = np.array([[0, -wx, -wy, -wz],
                          [wx, 0, wz, -wy],
                          [wy, -wz, 0, wx],
                          [wz, wy, -wx, 0]], dtype='float32')
        dq = 1/2*np.dot(omega,q)

        return dq

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
                parameters.adaptiveThreshConstant = 20

                parameters.cornerRefinementWinSize = 5
                parameters.cornerRefinementMethod = cv.aruco.CORNER_REFINE_CONTOUR
                # parameters.cornerRefinementMaxIterations = 20
                parameters.cornerRefinementMinAccuracy = 0.1
                # parameters = self.detector_aruco_parameters(10, 23, 10, 7)

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
                    # print("Marker Detected")

                    for i in range(0, len(markerIDs)):
                        # print("---------")
                        # print(i)

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
                            # print("Euler Reference:")
                            # print(euler_ref)

                            cv.aruco.drawAxis(image, self.mtx1, self.dist1, rvec_ref, tvec_ref, 0.5)
                            cv.aruco.drawDetectedMarkers(image, markerCorners[0])
        
                        
                        if markerIDs[i]==4:

                            # print("Marker Corners Movement:")
                            # print(markerCorners[0][i])

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
                            roll_mov, pitch_mov, yaw_mov = self.computeAngles(q_mov[3], q_mov[0], q_mov[1], q_mov[2])
                            euler_mov = np.array([roll_mov, pitch_mov, yaw_mov])
                            # print("Euler Movement")
                            # print(euler_mov)

                            if T_cr is not None:
                                #Homogeneous Transformation Object Frame to Fixed Frame
                                T_dr = T_cr@T_dc
                            else:
                                T_dr = np.eye(4)@T_dc
      
                            #Getting quaternions from rotation matrix
                            r_obj = sci.spatial.transform.Rotation.from_matrix(T_dr[0:3, 0:3])
                            q_obj = r_obj.as_quat()
                            # print(q_obj)
                            roll_obj, pitch_obj, yaw_obj = self.computeAngles(q_obj[3], q_obj[0], q_obj[1], q_obj[2])
                            euler_obj = np.array([roll_obj, pitch_obj, yaw_obj])
                            
                            #Real Quadcopter Attitude
                            attitude_real = self.quad_position.env.mat_rot
                            r_real = sci.spatial.transform.Rotation.from_matrix(attitude_real)
                            q_real = r_real.as_quat()
                            # q_real = self.quad_position.env.state[6:10]
                            # print(q_real)
                            roll_real, pitch_real, yaw_real = self.computeAngles(q_real[3], q_real[0], q_real[1], q_real[2])
                            # print("Attitude Real:{0}, {1}, {2}".format(roll_real, pitch_real, yaw_real))


                            #Marker's Position
                            zf_obj = float(T_dr[2,3])*0.868 - 2.52
                            xf_obj = float(T_dr[0,3])*0.978 - 0.0563
                            yf_obj = float(T_dr[1,3])*0.932 + 0.725

                            # xz = -9.37*10**-3*zf_obj + 0.018
                            # yz = 0.0102*zf_obj + -0.0208

                            # xf_obj -= xz
                            # yf_obj -= yz

                            #Measurement Vector
                            yaw_obj *= np.pi/180
                            pitch_obj *= np.pi/180
                            # cy = np.cos(yaw_obj)
                            # cx = np.sin(yaw_obj)
                            # cz = 0
                            # self.cam_vec = np.array([[cx, cy, cz]]).T
                            # self.cam_vec /= np.linalg.norm(self.cam_vec)

                            cx = np.sin(yaw_obj)*np.cos(pitch_obj)
                            cy = np.cos(yaw_obj)
                            cz = np.sin(yaw_obj)*np.sin(pitch_obj)
                            self.cam_vec = np.array([[cx, cy, cz]]).T
                            self.cam_vec *= 1/(np.linalg.norm(self.cam_vec))
                            # print(self.cam_vec)
                            

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


                            modulo = np.sqrt(q_obj[0]**2 + q_obj[1]**2 + q_obj[2]**2 + q_obj[3]**2)
                            #Print position values in frame
                            cv.putText(image, "X:"+str(np.round(float(xf_obj), 4)), (10,500), font, 1, (255,255,0), 2)
                            cv.putText(image, "Y:"+str(np.round(float(yf_obj), 4)), (100,500), font, 1, (255,255,0), 2)
                            cv.putText(image, "Z:"+str(np.round(float(zf_obj), 4)), (190,500), font, 1, (255,255,0), 2)
                            
                            cv.putText(image, "Orientacao Estimada por Camera:", (10, 20), font, 1, (255, 255, 255), 2)
                            cv.putText(image, "Phi:"+str(np.round(float(euler_obj[0]), 2)), (10,40), font, 1, (0,0,255), 2)
                            cv.putText(image, "Theta:"+str(np.round(float(euler_obj[1]), 2)), (10,60), font, 1, (0,255,0), 2)
                            cv.putText(image, "Psi:"+str(np.round(float(euler_obj[2]), 2)), (10,80), font, 1, (0,255,0), 2)

                            cv.putText(image, "Orientacao Real:", (10, 120), font, 1, (255, 255, 255), 2)
                            cv.putText(image, "Phi:"+str(np.round(float(roll_real), 2)), (10,140), font, 1, (0,0,255), 2)
                            cv.putText(image, "Theta:"+str(np.round(float(pitch_real), 2)), (10,160), font, 1, (0,255,0), 2)
                            cv.putText(image, "Psi:"+str(np.round(float(yaw_real), 2)), (10,180), font, 1, (255,0,0), 2)
                            
                            cv.putText(image, "q0:"+str(np.round(float(q_obj[3]), 3)), (10,400), font, 1, (255,255,255), 2)
                            cv.putText(image, "q1:"+str(np.round(float(q_obj[0]), 3)), (10,420), font, 1, (255,255,255), 2)
                            cv.putText(image, "q2:"+str(np.round(float(q_obj[1]), 3)), (10,440), font, 1, (255,255,255), 2)
                            cv.putText(image, "q3:"+str(np.round(float(q_obj[2]), 3)), (10,460), font, 1, (255,255,255), 2)
                            cv.putText(image, "Modulo:" + str(np.round(float(modulo), 3)), (10, 480), font, 1, (255, 0, 0),2)

                            self.angle_data.write("{:.2f} , ".format(float(euler_obj[0])) + "{:.2f} , ".format(float(euler_obj[1])) + "{:.2f} , ".format(float(euler_obj[2]))
                                + "{:.2f} , ".format(float(roll_real)) + "{:.2f} , ".format(float(pitch_real)) + "{:.2f} , ".format(float(yaw_real)) + "{:.2f}".format(self.index) + "\n")
                            self.pos_data.write("{:.4f} , ".format(float(xf_obj)) + "{:.4f} , ".format(float(yf_obj)) + "{:.4f} , ".format(float(zf_obj))
                                + "{:.4f} , ".format(float(x_real)) + "{:.4f} , ".format(float(y_real)) + "{:.4f} , ".format(float(z_real)) + "{:.2f}".format(self.index) + "\n")

                # print("Inner", self.index)
                self.index += 1
                
                cv.imshow('Drone Camera',image)
                cv.waitKey(1)
                self.angle_data.close()
                self.pos_data.close()

        return task.cont