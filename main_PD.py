import sys, os
import time

#Panda 3D Imports
from panda3d.core import Filename
from direct.showbase.ShowBase import ShowBase
from panda3d.core import loadPrcFile
loadPrcFile('./config/conf.prc')

import numpy as np
from models.world_setup import world_setup, quad_setup
from computer_vision.cameras_setup import cameras
from environment.position2 import quad_sim
from computer_vision.quadrotor_cv import computer_vision
import matplotlib.pyplot as plt
from models.camera_control import camera_control


mydir = os.path.abspath(sys.path[0])

mydir = Filename.fromOsSpecific(mydir).getFullpath()



frame_interval = 1
cam_names = ('cam_1', 'cam_2')



class MyApp(ShowBase):
    def __init__(self):

        ShowBase.__init__(self)     
        render = self.render


        # MODELS SETUP
        world_setup(self, render, mydir)
        quad_setup(self, render, mydir)

        self.quad_sim = quad_sim(self)            
        

        self.taskMgr.add(self.controller_function, 'Controller Read')

        # CAMERA NEUTRAL POSITION
        self.buffer_cameras = cameras(self, frame_interval, cam_names) 

        self.cv = computer_vision(self, self.quad_model, self.buffer_cameras.opencv_cameras[0], self.buffer_cameras.opencv_cameras[1], self.buffer_cameras.opencv_cam_cal[0], self.buffer_cameras.opencv_cam_cal[1], self.quad_sim)

        # CAMERA CONTROL
        camera_control(self, self.render)        

    
    def controller_function(self, task):

        init_time = time.time()

        self.cv.img_show(task.frame)

        if task.frame < self.quad_sim.outer_length:
            if task.frame==0:

                #Reset
                self.pos, self.vel, self.ang, self.ang_vel = self.quad_sim.quad_reset_random()
                self.pos_ant = self.pos.T
                self.vel_ant = self.vel
                
            else:
                
                cam_vec = self.cv.cam_vec
                cam_vec2 = self.cv.cam_vec2

                # self.pos = self.cv.position_mean

                # cam_vec = None
                # cam_vec2 = None

                #Step control
                pos, vel = self.quad_sim.step(cam_vec, cam_vec2, self.cv, task.frame)
                # self.pos = np.array([pos[0], self.pos_est[1], self.pos_est[2]])
                # self.vel = vel
            

                if task.frame % 100 == 0:
                    print('Step:', task.frame)
 
            return task.cont

        else:
            
            t = np.arange(0, self.quad_sim.time[-1], self.quad_sim.step_controller)

            #Plot Attitude
            
            fig, (a, b, c) = plt.subplots(3, 1, figsize=(10, 10))

            a.plot(t, self.quad_sim.phi_est_list, 'r', label=r'$\phi_{est}$')
            a.plot(t, self.quad_sim.phi_real_list, 'y--', label=r'$\phi_{real}$')
            a.plot(t, self.quad_sim.phi_des_list, 'g--', label=r'$\phi_{des}$')
            a.grid()
            a.legend()
            # a.set_ylim(-0.1, 0.1)

            b.plot(t, self.quad_sim.theta_est_list, 'r', label=r'$\theta_{est}$')
            b.plot(t, self.quad_sim.theta_real_list, 'y--', label=r'$\theta_{real}$')
            b.plot(t, self.quad_sim.theta_des_list, 'g--', label=r'$\theta_{des}$')
            b.grid()
            b.legend()
            # b.set_ylim(-0.1, 0.1)

            c.plot(t, self.quad_sim.psi_est_list, 'r', label=r'$\psi_{est}$')
            c.plot(t, self.quad_sim.psi_real_list, 'y--', label=r'$\psi_{real}$')
            c.plot(t, self.quad_sim.psi_ref, 'g--', label=r'$\psi_{des}$')
            c.grid()
            c.legend()
            # c.set_ylim(-1.6, 1.6)

            #------------------------------- Plot Position -----------------------------------------------

            fig2, (x, y, z) = plt.subplots(3, 1, figsize=(10, 10))

            x.plot(t, self.quad_sim.x_est_list, 'r', label=r'$x_{est}$')
            x.plot(t, self.quad_sim.x_real_list, 'y--', label=r'$x_{real}$')
            x.plot(t, self.quad_sim.x_ref, 'g--', label=r'$x_{des}$')
            x.grid()
            x.legend()
            # a.set_ylim(-0.1, 0.1)

            y.plot(t, self.quad_sim.y_est_list, 'r', label=r'$y_{est}$')
            y.plot(t, self.quad_sim.y_real_list, 'y--', label=r'$y_{real}$')
            y.plot(t, self.quad_sim.y_ref, 'g--', label=r'$y_{des}$')
            y.grid()
            y.legend()
            # b.set_ylim(-0.1, 0.1)

            z.plot(t, self.quad_sim.z_est_list, 'r', label=r'$z_{est}$')
            z.plot(t, self.quad_sim.z_real_list, 'y--', label=r'$z_{real}$')
            z.plot(t, self.quad_sim.z_ref, 'g--', label=r'$z_{des}$')
            z.grid()
            z.legend()
            # c.set_ylim(-1.6, 1.6)

            plt.show()

            return task.done
           

app = MyApp()

app.run()
    