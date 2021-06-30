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
import math
from sklearn.metrics import mean_squared_error
from mpl_toolkits import mplot3d


mydir = os.path.abspath(sys.path[0])

mydir = Filename.fromOsSpecific(mydir).getFullpath()



frame_interval = 5
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
        

        if task.frame < self.quad_sim.outer_length:

            if task.frame==0:

                #Reset
                self.quad_sim.quad_reset_random()
                
            else:
                
                # self.cv.img_show(task.frame)

                #Step control
                self.quad_sim.step(self.cv, task.frame)
            

                if task.frame % 100 == 0:
                    print('Step:', task.frame)
 
            return task.cont

        else:
            
            t = np.arange(0, self.quad_sim.time[-1], self.quad_sim.step_controller)

            #Plot Attitude
            
            fig, (a, b, c) = plt.subplots(3, 1, figsize=(10, 10), sharex = True)
            a.set_title('Atitude')

            a.plot(t, self.quad_sim.phi_est_list, 'r', label=r'$\phi_{est}$')
            # a.plot(t, self.quad_sim.phi_real_list, 'y--', label=r'$\phi_{real}$')
            a.plot(t, self.quad_sim.phi_des_list, 'g--', label=r'$\phi_{des}$')
            a.grid()
            a.set_ylabel(r'$\phi$ (rad)')
            a.legend()
            # a.set_ylim(-0.1, 0.1)

            b.plot(t, self.quad_sim.theta_est_list, 'r', label=r'$\theta_{est}$')
            # b.plot(t, self.quad_sim.theta_real_list, 'y--', label=r'$\theta_{real}$')
            b.plot(t, self.quad_sim.theta_des_list, 'g--', label=r'$\theta_{des}$')
            b.grid()
            b.set_ylabel(r'$\theta$ (rad)')
            b.legend()
            # b.set_ylim(-0.1, 0.1)

            c.plot(t, self.quad_sim.psi_est_list, 'r', label=r'$\psi_{est}$')
            # c.plot(t, self.quad_sim.psi_real_list, 'y--', label=r'$\psi_{real}$')
            c.plot(t, self.quad_sim.psi_ref, 'g--', label=r'$\psi_{des}$')
            c.plot(self.quad_sim.time, self.quad_sim.psi_wp, 'bo', label=r'$\psi_{waypoint}$')
            c.grid()
            c.set_ylabel(r'$\psi$ (rad)')
            c.set_xlabel('Tempo (s)')
            c.legend()
            # c.set_ylim(-1.6, 1.6)

            #------------------------------- Plot Position -----------------------------------------------

            fig2, (x, y, z) = plt.subplots(3, 1, figsize=(10, 10), sharex = True)
            x.set_title('Posição')

            x.plot(t, self.quad_sim.x_est_list, 'r', label=r'$x_{est}$')
            # x.plot(t, self.quad_sim.x_real_list, 'y--', label=r'$x_{real}$')
            x.plot(t, self.quad_sim.x_ref, 'g--', label=r'$x_{des}$')
            x.plot(self.quad_sim.time, self.quad_sim.x_wp, 'bo', label=r'$x_{waypoint}$')
            x.grid()
            x.set_ylabel(r'$X$ (m)')
            x.legend()
            # a.set_ylim(-0.1, 0.1)

            y.plot(t, self.quad_sim.y_est_list, 'r', label=r'$y_{est}$')
            # y.plot(t, self.quad_sim.y_real_list, 'y--', label=r'$y_{real}$')
            y.plot(t, self.quad_sim.y_ref, 'g--', label=r'$y_{des}$')
            y.plot(self.quad_sim.time, self.quad_sim.y_wp, 'bo', label=r'$y_{waypoint}$')
            y.grid()
            y.set_ylabel(r'$Y$ (m)')
            y.legend()
            # b.set_ylim(-0.1, 0.1)

            z.plot(t, self.quad_sim.z_est_list, 'r', label=r'$z_{est}$')
            # z.plot(t, self.quad_sim.z_real_list, 'y--', label=r'$z_{real}$')
            z.plot(t, self.quad_sim.z_ref, 'g--', label=r'$z_{des}$')
            z.plot(self.quad_sim.time, self.quad_sim.z_wp, 'bo', label=r'$z_{waypoint}$')
            z.grid()
            z.set_ylabel(r'$Z$ (m)')
            z.set_xlabel('Tempo (s)')
            z.legend()
            # c.set_ylim(-1.6, 1.6)

            #-------------------------------------- Plot Trajectory ----------------------------------------------------------

            fig3 = plt.figure()
            ax = plt.axes(projection='3d')
            ax.set_title('Trajetória')
            ax.plot3D(self.quad_sim.x_ref, self.quad_sim.y_ref, self.quad_sim.z_ref, 'g--')
            # ax.plot3D(self.quad_sim.x_real_list, self.quad_sim.y_real_list, self.quad_sim.z_real_list, 'y--')
            ax.plot3D(self.quad_sim.x_est_list, self.quad_sim.y_est_list, self.quad_sim.z_est_list, 'r')
            ax.scatter(self.quad_sim.x_wp, self.quad_sim.y_wp, self.quad_sim.z_wp, 'bo')

            plt.show()


            #Mean squared error w.r.t desired state
            phi_mse = mean_squared_error(self.quad_sim.phi_des_list, self.quad_sim.phi_est_list)
            theta_mse = mean_squared_error(self.quad_sim.theta_des_list, self.quad_sim.theta_est_list)
            psi_mse = mean_squared_error(self.quad_sim.psi_ref, self.quad_sim.psi_est_list)

            x_mse = mean_squared_error(self.quad_sim.x_ref, self.quad_sim.x_est_list)
            y_mse = mean_squared_error(self.quad_sim.y_ref, self.quad_sim.y_est_list)
            z_mse = mean_squared_error(self.quad_sim.z_ref, self.quad_sim.z_est_list)

            #Mean squared error w.r.t desired state
            # phi_mse = mean_squared_error(self.quad_sim.phi_real_list, self.quad_sim.phi_est_list)
            # theta_mse = mean_squared_error(self.quad_sim.theta_real_list, self.quad_sim.theta_est_list)
            # psi_mse = mean_squared_error(self.quad_sim.psi_real_list, self.quad_sim.psi_est_list)

            # x_mse = mean_squared_error(self.quad_sim.x_real_list, self.quad_sim.x_est_list)
            # y_mse = mean_squared_error(self.quad_sim.y_real_list, self.quad_sim.y_est_list)
            # z_mse = mean_squared_error(self.quad_sim.z_real_list, self.quad_sim.z_est_list)

            #RMSE
            phi_rmse = math.sqrt(phi_mse)
            theta_rmse = math.sqrt(theta_mse)
            psi_rmse = math.sqrt(psi_mse)

            x_rmse = math.sqrt(x_mse)
            y_rmse = math.sqrt(y_mse)
            z_rmse = math.sqrt(z_mse)

            print("MEKF Attitude RMSE: \n Phi - {0} \n Theta - {1} \n Psi - {2}".format(phi_rmse, theta_rmse, psi_rmse))

            print("Posição RMSE: \n X - {0} \n Y - {1} \n Z - {2}".format(x_rmse, y_rmse, z_rmse))



            return task.done
           

app = MyApp()

app.run()
    