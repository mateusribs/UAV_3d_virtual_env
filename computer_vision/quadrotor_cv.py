import cv2 as cv
import numpy as np
from collections import deque
import math
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
        self.cv_cam.cam.node().getLens().setFilmSize(36,24)
        self.cv_cam.cam.node().getLens().setFocalLength(45)
        self.cv_cam.cam.setPos(0, 0, 6.5)
        self.cv_cam.cam.setHpr(360, 270, 0)
        self.cv_cam.cam.reparentTo(self.render.render)
        
        # r = 0.025
        # self.objp = np.array([[0, 0, 0.06],[0 + r, 0, 0.06], [0 - r, 0, 0.06], [0, 0 + r, 0.06], [0, 0 - r, 0.06], [0 + r/2, 0, 0.06], [0 - r/2, 0, 0.06], [0, 0 + r/2, 0.06], [0, 0 - r/2, 0.06], [0 + r/10, 0, 0.06], [0 - r/10, 0, 0.06], [0, 0 + r/10, 0.06], [0, 0 - r/10, 0.06],
        #                      [0, 0.15, 0.04], [0 + r, 0.15, 0.04], [0 - r, 0.15, 0.04], [0, 0.15 + r, 0.04], [0, 0.15 - r, 0.04], [0 + r/2, 0.15, 0.04], [0 - r/2, 0.15, 0.04], [0, 0.15 + r/2, 0.04], [0, 0.15 - r/2, 0.04], [0 + r/10, 0.15, 0.04], [0 - r/10, 0.15, 0.04], [0, 0.15 + r/10, 0.04], [0, 0.15 - r/10, 0.04],
        #                      [0.15, 0, 0.04], [0.15 + r, 0, 0.04], [0.15 - r, 0, 0.04], [0.15, 0 + r, 0.04], [0.15, 0 - r, 0.04], [0.15 + r/2, 0, 0.04], [0.15 - r/2, 0, 0.04], [0.15, 0 + r/2, 0.04], [0.15, 0 - r/2, 0.04], [0.15 + r/10, 0, 0.04], [0.15 - r/10, 0, 0.04], [0.15, 0 + r/10, 0.04], [0.15, 0 - r/10, 0.04],
        #                      [0, -0.15, 0.04], [0 + r, -0.15, 0.04], [0 - r, -0.15, 0.04], [0, -0.15 + r, 0.04], [0, -0.15 - r, 0.04], [0 + r/2, -0.15, 0.04], [0 - r/2, -0.15, 0.04], [0, -0.15 + r/2, 0.04], [0, -0.15 - r/2, 0.04], [0 + r/10, -0.15, 0.04], [0 - r/10, -0.15, 0.04], [0, -0.15 + r/10, 0.04], [0, -0.15 - r/10, 0.04]], dtype=np.float32)
        
        # self.objp = np.array([[0, 0, 0.06],
        #                      [0, 0.15, 0.04],
        #                      [0.15, 0, 0.04],
        #                      [0, 0.15, 0.04]], dtype = np.float32)
        

        self.obj_frame = []
        self.ground_frame = []
        self.distances = []
        self.T_flag = False

        self.render.taskMgr.add(self.img_show, 'OpenCv Image Show')


    def detect_contourn(self, image, color):
        
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

        if color == "Red":
            #Define the limits in HSV variables
            self.lower = np.array([0, 35, 225])
            self.upper = np.array([0, 255, 255])
        if color == "Green":
            #Define the limits in HSV variables
            self.lower = np.array([48, 35, 225])
            self.upper = np.array([65, 255, 255])
        if color == "Blue":
            #Define the limits in HSV variables
            self.lower = np.array([70, 35, 225])
            self.upper = np.array([120, 255, 255])
        if color == "Yellow":
            #Define the limits in HSV variables
            self.lower = np.array([20, 100, 100])
            self.upper = np.array([32, 220, 255])
        
        # l_h = cv.getTrackbarPos("L - H", "Trackbars")
        # l_s = cv.getTrackbarPos("L - S", "Trackbars")
        # l_v = cv.getTrackbarPos("L - V", "Trackbars")
        # u_h = cv.getTrackbarPos("U - H", "Trackbars")
        # u_s = cv.getTrackbarPos("U - S", "Trackbars")
        # u_v = cv.getTrackbarPos("U - V", "Trackbars")  

        # self.lower = np.array([l_h, l_s, l_v])
        # self.upper = np.array([u_h, u_s, u_v])

        #Define threshold for red color
        mask = cv.inRange(hsv, self.lower, self.upper)
        #Create a kernel
        kernel = np.ones((5,5), np.uint8)
        #Apply opening process
        opening = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations = 1)
        #Find BLOB's contours
        cnts, _ = cv.findContours(opening.copy(), cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

        return  cnts

    def center_mass_calculate(self, image, c):

        # Compute the center of the contour
        M = cv.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        _, radius = cv.minEnclosingCircle(c)
        cX = int(cX)
        cY = int(cY)
        center = (cX, cY)
        radius = int(radius)
        perimeter = cv.arcLength(c, True)
        #Compute the eccentricity
        metric = (4*math.pi*M["m00"])/perimeter**2
        if metric > 0.8:
            #Draw the contour and center of the shape on the image
            cv.drawContours(image, [c], -1, (0, 0, 0), 1)
            # cv.circle(image, center, radius, (0, 0, 0),1)
            cv.circle(image, center, 1, (0, 0, 0), -1)
            # cv.circle(image, (cX+radius, cY), 1, (0, 0, 0), 2)
            # cv.circle(image, (cX-radius, cY), 1, (0, 0, 0), 2)
            # cv.circle(image, (cX, cY+radius), 1, (0, 0, 0), 2)
            # cv.circle(image, (cX, cY-radius), 1, (0, 0, 0), 2)
                          
        return cX, cY, radius

    def detect_corners(self, ret, image):

        if ret:
            img = cv.cvtColor(image, cv.COLOR_RGBA2BGR)
            self.gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)              
            ret, corners = cv.findChessboardCorners(self.gray, (self.nCornersCols, self.nCornersRows), 
                                                        cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FILTER_QUADS+ cv.CALIB_CB_FAST_CHECK)
            
            return ret, corners

    def draw(self, image, corners, imgpts):
        corner = tuple(corners[0].ravel())
        img = cv.line(image, corner, tuple(imgpts[0].ravel()), (255,0,0), 3)
        img = cv.line(image, corner, tuple(imgpts[1].ravel()), (0,255,0), 3)
        img = cv.line(image, corner, tuple(imgpts[2].ravel()), (0,255,255), 3)
        return img
    
    def get_element_vector(self, f1, f2, c1, c2):
        #Where f1 is frameA, f2 is frameB
        #c1 is the coordinate, let x = 0, y = 1, z = 2
        vec = []
        for i in range(np.shape(f1)[0]):
            cc = f2[i, c1]*f1[i, c2]
            vec.append(cc)
        return np.sum(vec)

    def get_element_A(self, f1, c1, c2):
        A = []
        for i in range(np.shape(f1)[0]):
            cc = f1[i, c1]*f1[i, c2]
            A.append(cc)
        return np.sum(A)

    def get_element_last(self, f1, c1):
        last = []
        for i in range(np.shape(f1)[0]):
            cc = f1[i, c1]
            last.append(cc)
        return np.sum(last)

    def get_transform_frame(self, f1, f2):
        matrix = np.zeros((3,4))
        for i in range(3):
            for j in range(3):
                matrix[i, j] = self.get_element_vector(f1, f2, i, j)
                matrix[i, 3] = self.get_element_last(f2, i)

        A = np.zeros((4,4))
        for i in range(3):
            for j in range(3):
                A[i, j] = self.get_element_A(f1, i, j)

        for i in range(3):
            A[i,3] = self.get_element_last(f1, i)
            A[3, i] = self.get_element_last(f1, i)

        A[3,3] = np.shape(f1)[0]
        A_inv = np.linalg.inv(A)

        matrix = np.transpose(matrix)

        T = np.dot(A_inv, matrix)
        T = np.transpose(T)
        last_row = np.array([0,0,0,1]).reshape(1,4)
        T = np.concatenate((T, last_row), axis=0)
        
        return T

    def get_pose(self, image, objpoints, imgpoints, mtx, dist):
        
        axis = np.float32([[.05, 0, 0], [0, .05, 0], [0, 0, 0.05]]).reshape(-1,3)

        # Find the rotation and translation vectors.
        _, rvecs, tvecs,_ = cv.solvePnPRansac(objpoints, imgpoints, mtx, dist, iterationsCount=5)
        # project 3D points to image plane
        imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
        imgpts = imgpts.astype(np.int)
        img = self.draw(image, imgpoints, imgpts)
        R_matrix, _ = cv.Rodrigues(rvecs)
        
        return rvecs, tvecs, R_matrix, image

    def detector_aruco_parameters(self, min, max, step, const):
        #This function will set the detection parameters of aruco markers

        parameters = cv.aruco.DetectorParameters_create()
        parameters.adaptiveThreshWinSizeMin = 10
        parameters.adaptiveThreshWinSizeMax = 23
        parameters.adaptiveThreshWinSizeStep = 10
        parameters.adaptiveThreshConstant = 7

        return parameters

    def print_pose(self, image, orientation, translation):

        font = cv.FONT_HERSHEY_PLAIN
        cv.putText(image, "Roll:"+str(round(orientation[0], 2)), (10,20), font, 1, (0,255,0), 2)
        cv.putText(image, "Pitch:"+str(round(orientation[1], 2)), (10,40), font, 1, (0,255,0), 2)
        cv.putText(image, "Yaw:"+str(round(orientation[2], 2)), (10,60), font, 1, (0,255,0), 2)

        cv.putText(image, "X:"+str(round(translation[0], 2)), (10,80), font, 1, (255,0,0), 2)
        cv.putText(image, "Y:"+str(round(translation[1], 2)), (10,100), font, 1, (255,0,0), 2)
        cv.putText(image, "Z:"+str(round(translation[2], 2)), (10,120), font, 1, (255,0,0), 2)

    def rot2euler(self, rot):
        
        phi = 180*math.atan2(rot[2,1], rot[2,2])/math.pi
        theta = 180*math.atan2(-rot[2, 0], math.sqrt(rot[2,1]**2 + rot[2,2]**2))/math.pi
        psi = 180*math.atan2(rot[1,0], rot[0,0])/math.pi

        orientation = np.array([phi, theta, psi])

        return orientation


    def img_show(self, task):

        cX1 = 0
        cY1 = 0
        cX2 = 0
        cY2 = 0
        cX3 = 0
        cY3 = 0
        cX4 = 0
        cY4 = 0
        r1 = 0
        r2 = 0
        r3 = 0
        r4 = 0

        global tvecs
        #Font setup
        font = cv.FONT_HERSHEY_PLAIN
        
        #Load the predefinied dictionary
        dictionary = cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_250)

        if task.frame % self.cv_cam.frame_int == 1:           
            ret, image = self.cv_cam.get_image()
            #ret, image2 = self.cv_cam_2.get_image()

            ##Chessboard 
            if ret:

################################# CHESSBOARD POSE ESTIMATION ##############################################################################

                # ret, corners = self.detect_corners(ret, image)
                # if ret:                
                #     if len(corners)==54:
                #         rvecs, tvecs, R_matrix, image = self.get_pose(image, self.objp, corners, self.mtx1, self.dist1, self.rvec, self.tvec)
                        
                #         print("R - PNP: ", R_matrix)
                #         tvec = np.concatenate((tvecs, np.ones((1,1))), axis=0)
                #         #print("Position:" ,self.quad_position.env.state[0:5:2])
                #         obj_pos = np.reshape(tvecs, (1,3))
                #         obj_pos = np.asarray(obj_pos, np.float32)
                #         self.obj_frame.append(obj_pos)
                
                #         ground_pos = np.reshape(self.quad_position.env.state[0:5:2], (1,3))
                #         ground_pos = np.asarray(ground_pos, np.float32)
                #         self.ground_frame.append(ground_pos)

                #         if len(self.obj_frame)==1 and len(self.ground_frame)==1:

                #             global T

                #             self.obj_frame = np.asarray(self.obj_frame, np.float32).reshape(1,3)
                #             self.ground_frame = np.asarray(self.ground_frame, np.float32).reshape(1,3)

                #             T = self.get_transform_frame(self.obj_frame, self.ground_frame)

                #             #euler angles from quadrotor
                #             quad_rot = self.quad_position.env.mat_rot
                #             phi_quad = 180*math.atan2(quad_rot[2,1], quad_rot[2,2])/math.pi
                #             theta_quad = 180*math.atan2(-quad_rot[2,0], math.sqrt(quad_rot[0,0]**2 + quad_rot[0,1]**2))/math.pi
                #             psi_quad = 180*math.atan2(quad_rot[1,0], quad_rot[0,0])/math.pi
                #             print("Quadrotor Angles")
                #             print("Roll:",phi_quad," Pitch:",theta_quad," Yaw:",psi_quad)
                #             #estimated angles from camera
                #             phi_est = 180*math.atan2(R_matrix[2,1], R_matrix[2,2])/math.pi
                #             theta_est = 180*math.atan2(-R_matrix[2,0], math.sqrt(R_matrix[0,0]**2 + R_matrix[0,1]**2))/math.pi
                #             psi_est = 180*math.atan2(R_matrix[1,0], R_matrix[0,0])/math.pi
                #             print("Estimated Angles")
                #             print("Roll:",phi_est," Pitch:",theta_est," Yaw:",psi_est)
                #             print("Matriz Rotação Original:", quad_rot)
                #             print("Matriz Rotação Estimada:", R_matrix)

                #             self.T_flag = True
                #             self.obj_frame = []
                #             self.ground_frame = []

                #         if self.T_flag:
                #             real_pos = np.dot(T, tvec)
                #             erro_X = self.quad_position.env.state[0] - real_pos[0]
                #             erro_Y = self.quad_position.env.state[2] - real_pos[1]
                #             erro_Z = self.quad_position.env.state[4] - real_pos[2]
                #             # print("Ground Frame:", self.quad_position.env.state[0:5:2])
                #             # print("Real Frame:", np.transpose(real_pos)[:,:3])
                #             # print("E_X:",erro_X," E_Y:",erro_Y," E_Y:", erro_Z)
                            
                #         cv.putText(image, "X:"+str(tvecs[0])+"Y:"+str(tvecs[1])+ "Z:"+str(tvecs[2])
                #         , (10,10), font, 1, (255, 255, 255), 1)


###################################### CIRCLE MARKERS POSE ESTIMATION #####################################################################

                # cnts_red = self.detect_contourn(image, "Red")
                # cnts_green = self.detect_contourn(image, "Green")
                # cnts_blue = self.detect_contourn(image, "Blue")
                # cnts_yellow = self.detect_contourn(image, "Yellow")
                # # loop over the contours
                # for c in cnts_red:
                #     cX1, cY1, r1 = self.center_mass_calculate(image, c)
                # for c in cnts_green:
                #     cX2, cY2, r2 = self.center_mass_calculate(image, c)
                # for c in cnts_blue:
                #     cX3, cY3, r3 = self.center_mass_calculate(image, c)
                # for c in cnts_yellow:
                #     cX4, cY4, r4 = self.center_mass_calculate(image, c)


                # imgpoints = np.array([[cX1, cY1], [cX1+r1, cY1], [cX1-r1, cY1],[cX1, cY1+r1], [cX1, cY1-r1], [cX1+r1/2, cY1], [cX1-r1/2, cY1],[cX1, cY1+r1/2], [cX1, cY1-r1/2], [cX1+r1/10, cY1], [cX1-r1/10, cY1],[cX1, cY1+r1/10], [cX1, cY1-r1/10],
                #                     [cX2, cY2], [cX2+r2, cY2], [cX2-r2, cY2], [cX2, cY2+r2], [cX2, cY2-r2], [cX2+r2/2, cY2], [cX2-r2/2, cY2], [cX2, cY2+r2/2], [cX2, cY2-r2/2], [cX2+r2/10, cY2], [cX2-r2/10, cY2], [cX2, cY2+r2/10], [cX2, cY2-r2/10],
                #                     [cX3, cY3], [cX3+r3, cY3], [cX3-r3, cY3], [cX3, cY3+r3], [cX3, cY3-r3], [cX3+r3/2, cY3], [cX3-r3/2, cY3], [cX3, cY3+r3/2], [cX3, cY3-r3/2], [cX3+r3/10, cY3], [cX3-r3/10, cY3], [cX3, cY3+r3/10], [cX3, cY3-r3/10],
                #                     [cX4, cY4], [cX4+r4, cY4], [cX4-r4, cY4], [cX4, cY4+r4], [cX4, cY4-r4], [cX4+r4/2, cY4], [cX4-r4/2, cY4], [cX4, cY4+r4/2], [cX4, cY4-r4/2], [cX4+r4/10, cY4], [cX4-r4/10, cY4], [cX4, cY4+r4/10], [cX4, cY4-r4/10]], np.float32)
                

                # # imgpoints = np.array([[cX1, cY1], [cX2, cY2], [cX3, cY3], [cX4, cY4]], np.float32)


                # if cX1 != 0 and cY1 != 0:
                #     global R_matrix

                #     rvecs, tvecs, R_matrix, image = self.get_pose(image, self.objp, imgpoints, self.mtx1, self.dist1)                    
                #     tvec = np.concatenate((tvecs, np.ones((1,1))), axis=0)
                #     #print("Position:" ,self.quad_position.env.state[0:5:2])
                #     obj_pos = np.reshape(tvecs, (1,3))
                #     obj_pos = np.asarray(obj_pos, np.float32)
                #     self.obj_frame.append(obj_pos)
                    
                #     ground_pos = np.reshape(self.quad_position.env.state[0:5:2], (1,3))
                #     ground_pos = np.asarray(ground_pos, np.float32)
                #     self.ground_frame.append(ground_pos)
                


                # if len(self.obj_frame)==1 and len(self.ground_frame)==1:

                #     global T

                #     self.obj_frame = np.asarray(self.obj_frame, np.float32).reshape(1,3)
                #     self.ground_frame = np.asarray(self.ground_frame, np.float32).reshape(1,3)

                #     T = self.get_transform_frame(self.obj_frame, self.ground_frame)
                #     self.T_flag = True
                #     self.obj_frame = []
                #     self.ground_frame = []
                #     #print("R: ", self.quad_position.env.mat_rot)

                #     #euler angles from quadrotor
                #     quad_rot = self.quad_position.env.mat_rot
                #     phi_quad = 180*math.atan2(-quad_rot[2,1], quad_rot[2,2])/math.pi
                #     theta_quad = 180*math.asin(quad_rot[2,0])/math.pi
                #     psi_quad = 180*math.atan2(-quad_rot[1,0], quad_rot[0,0])/math.pi
                #     print("Quadrotor Angles")
                #     print("Roll:",phi_quad," Pitch:",theta_quad," Yaw:",psi_quad)
                #     #estimated angles from camera
                #     phi_est = 180*math.atan2(-R_matrix[2,1], R_matrix[2,2])/math.pi
                #     theta_est = 180*math.asin(R_matrix[2, 0])/math.pi
                #     psi_est = 180*math.atan2(-R_matrix[1,0], R_matrix[0,0])/math.pi
                #     print("Estimated Angles")
                #     print("Roll:",phi_est," Pitch:",theta_est," Yaw:",psi_est)
                #     print("Matriz Rotação Original:", quad_rot)
                #     print("Matriz Rotação Estimada:", R_matrix)

                #     if self.T_flag:
                #         real_pos = np.dot(T, tvec)
                #         erro_X = self.quad_position.env.state[0] - real_pos[0]
                #         erro_Y = self.quad_position.env.state[2] - real_pos[1]
                #         erro_Z = self.quad_position.env.state[4] - real_pos[2]
                #         print("Ground Frame:", self.quad_position.env.state[0:5:2])
                #         print("Real Frame:", np.transpose(real_pos)[:,:3])
                #         # print("E_X:",erro_X," E_Y:",erro_Y," E_Y:", erro_Z)
                #         self.T_flag = False


                # # Print the image coordinates on the screen
                # cv.putText(image," Center:"+str(cX1)+','+str(cY1), (10, 10), font, 1, (255,0,0), 1)
                # cv.putText(image," Center:"+str(cX2)+','+str(cY2), (10, 25), font, 1, (0,255,0), 1)
                # cv.putText(image," Center:"+str(cX3)+','+str(cY3), (10, 40), font, 1, (0,0,255), 1)
                # cv.putText(image," Center:"+str(cX4)+','+str(cY4), (10, 55), font, 1, (0,255,255), 1)

####################################### ARUCO MARKER POSE ESTIMATION #########################################################################

                image = cv.cvtColor(image, cv.COLOR_BGRA2BGR)

                #Load the predefinied dictionary
                dictionary = cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_250)

                #Initialize the detector parameters using defaults values
                parameters = self.detector_aruco_parameters(10, 23, 10, 7)

                #Detect the markers in the image
                markerCorners, markerIDs, rejectedCandidates = cv.aruco.detectMarkers(image, dictionary, parameters=parameters)

                #If there is a marker compute the pose
                if np.all(markerIDs != None):
                    print("Marker Detected")
                    rvecs, tvecs, _ = cv.aruco.estimatePoseSingleMarkers(markerCorners, 0.1, self.mtx1, self.dist1)

                    #Convert rotation vector to rotation matrix
                    R_matrix, _ = cv.Rodrigues(rvecs)
                    #Translation vector
                    pos_meas = tvecs.ravel()

                    #Draw the pose estimated
                    for i in range(0, markerIDs.size):
                        cv.aruco.drawAxis(image, self.mtx1, self.dist1, rvecs[i], tvecs[i], 0.1)
                    
                    #Estimated states
                    #Convert rotation matrix to euler angles (ZYX sequence)
                    orientation_est = self.rot2euler(R_matrix)
                    #Print on the screen the orientation and translation values
                    self.print_pose(image, orientation_est, pos_meas)

                    #Real states
                    orientation_real = self.rot2euler(self.quad_position.env.mat_rot)
                    print("Real Orientation:", orientation_real)
                    print("Real Translation:", self.quad_position.env.state[0:5:2])


                cv.aruco.drawDetectedMarkers(image, markerCorners)                

                cv.imshow('Drone Camera',image)
                cv.waitKey(1)

        return task.cont