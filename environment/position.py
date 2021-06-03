import torch
import time
import numpy as np
import sys
from environment.quadrotor_env import quad, sensor
from environment.quaternion_euler_utility import deriv_quat
from environment.controller.model import ActorCritic
from environment.controller.dl_auxiliary import dl_in_gen
from mission_control.mission_control import mission
from environment.quadrotor_control import Controller

## PPO SETUP ##
time_int_step = 0.01
T = 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class quad_position():
    
    def __init__(self, render, quad_model, prop_models, EPISODE_STEPS, REAL_CTRL, ERROR_AQS_EPISODES, ERROR_PATH, HOVER, M_C):
        self.REAL_CTRL = True
        self.IMG_POS_DETER = False
        self.ERROR_AQS_EPISODES = ERROR_AQS_EPISODES
        self.ERROR_PATH = ERROR_PATH
        self.HOVER = HOVER
        self.M_C = M_C
        
        self.log_state = []
        self.log_target = []
        self.log_input = []

        self.quad_model = quad_model
        self.prop_models = prop_models
        self.episode_n = 1
        self.time_total_sens = []
        self.T = T
        
        self.render = render
        self.render.taskMgr.add(self.drone_position_task, 'Drone Position')
        
        # ENV SETUP
        self.env = quad(time_int_step, EPISODE_STEPS, direct_control=1, T=T)
        self.sensor = sensor(self.env)
        
        #CONTROLLER SETUP
        self.inner_length = 10
        self.controller = Controller(total_time = 10, sample_time = 0.01, inner_length = self.inner_length)
        
        # self.PD = True


        self.x_wp = np.array([[0, 1, 1, 0, 0]]).T
        self.y_wp = np.array([[0, 1, 1, 2, 2]]).T
        self.z_wp = np.array([[0, 1, 1.5, 2, 2.5]]).T
        self.psi_wp = np.array([[0, np.pi/12, np.pi/8, np.pi/4, np.pi]]).T
        self.time = [0, 2, 4, 6, 8]
        self.step_controller = 0.1

        
    def drone_position_task(self, task):

        if task.frame == 0 or self.env.done:

            #MISSION CONTROL SETUP
            # if self.M_C:
            #     self.mission_control = mission(time_int_step)
            #     # self.mission_control.sin_trajectory(4000, 1, 0.1, np.array([0, 0, 0]), np.array([1, 1, 0]))
            #     # self.mission_control.spiral_trajectory(4000, 0.5, np.pi/10, 1, np.array([0,0,0]))
            #     self.mission_control.gen_trajectory(4000, np.array([0, 0, 0]))
            #     # self.error_mission = np.zeros(14)
            # else:
            #     self.error_mission = np.zeros(14)
            # self.control_error_list = []
            
            # self.estimation_error_list = []
            # # if self.HOVER:
            # #     in_state = np.array([0, 0, 0, 0, 0, 0, 1, 0.000, 0.000, 0.000, 0, 0, 0])
            # if self.HOVER and self.episode_n==4:
            #     in_state = np.array([0, 0, 0.3, 0, 0, 0, 1.000, 0.000, 0.000, 0.000, 0, 0, 0])
            # elif self.HOVER and self.episode_n==3:
            #     in_state = np.array([0, 0, 0.3, 0, -2, 0, 1.000, 0.000, 0.000, 0.000, 0, 0, 0])
            # elif self.HOVER and self.episode_n==1:
            #     in_state = np.array([0.5, 0.0, 0.3, 0, -0.5, 0, 0.831, -0.212, 0.227, -0.462, 0, 0, 0])
            # elif self.HOVER and self.episode_n==2:
            #     in_state = np.array([-0.3, 0.0, 1.0, 0, -1, 0, 0.775, -0.342, -0.092, 0.525, 0, 0, 0])
            # else:
            #     in_state = None

            #Planner Trajectory
            _, _, x_matrix = self.controller.getCoeff_snap(self.x_wp, self.time)
            _, _, y_matrix = self.controller.getCoeff_snap(self.y_wp, self.time)
            _, _, z_matrix = self.controller.getCoeff_snap(self.z_wp, self.time)
            _, _, psi_matrix = self.controller.getCoeff_accel(self.psi_wp, self.time)

            self.x_ref, self.dotx_ref, self.ddotx_ref, _, _ = self.controller.evaluate_equations_snap(self.time, self.step_controller, x_matrix)
            self.y_ref, self.doty_ref, self.ddoty_ref, _, _ = self.controller.evaluate_equations_snap(self.time, self.step_controller, y_matrix)
            self.z_ref, self.dotz_ref, self.ddotz_ref, _, _ = self.controller.evaluate_equations_snap(self.time, self.step_controller, z_matrix)
            self.psi_ref, _, _ = self.controller.evaluate_equations_accel(self.time, self.step_controller, psi_matrix)

            self.outer_length = len(self.x_ref)
            #Initial State
            
            x0 = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])

            x_atual, _ = self.env.reset(x0)
            
            ang = self.quad_model.ang
            ang_vel = self.quad_model.ang_vel

            self.pos_atual = np.array([[x_atual[0,0], x_atual[0,2], x_atual[0,4]]]).T
            self.vel_atual = np.array([[x_atual[0,1], x_atual[0,3], x_atual[0,5]]]).T
            self.ang_atual = np.array([[ang[0], ang[1], ang[2]]]).T
            self.ang_vel_atual = np.array([[ang_vel[0], ang_vel[1], ang_vel[2]]]).T

            self.sensor.reset()
            pos = self.env.state[0:5:2]
            ang = self.env.ang
            self.a = np.zeros(4)
            self.episode_n += 1
            print(f'Episode Number: {self.episode_n}')
            
        else:
            #Position Control
            pos_ref = np.array([[self.x_ref[i], self.y_ref[i], self.z_ref[i]]]).T
            vel_ref = np.array([[self.dotx_ref[i], self.doty_ref[i], self.dotz_ref[i]]]).T
            accel_ref = np.array([[self.ddotx_ref[i], self.ddoty_ref[i], self.ddotz_ref[i]]]).T
            # print(pos_ref, pos_atual)
            psi = psi_ref[i]
            # print(quat_z)
            # T, phi_des, theta_des = controller.pos_control(pos_atual, pos_ref, vel_atual, vel_ref, accel_ref, psi, Kt)

            T, phi_des, theta_des = self.controller.pos_control_PD(pos_atual, pos_ref, vel_atual, vel_ref, accel_ref, psi)
            
            ang_des = np.array([[float(phi_des), float(theta_des), float(psi)]]).T

            for j in range(self.inner_length):

                # taux, tauy, tauz = controller.att_control(ang_atual, ang_des, ang_vel_atual, Ka)
                taux, tauy, tauz = self.controller.att_control_PD(ang_atual, ang_vel_atual, ang_des)

                action = np.array([float(T), taux, tauy, tauz])
                x, _, _ = self.quad_model.step(action)
                
                ang = self.quad_model.ang
                ang_vel = self.quad_model.ang_vel

                pos_atual = np.array([[x[0], x[2], x[4]]]).T
                vel_atual = np.array([[x[1], x[3], x[5]]]).T
                ang_atual = np.array([[ang[0], ang[1], ang[2]]]).T
                ang_vel_atual = np.array([[ang_vel[0], ang_vel[1], ang_vel[2]]]).T


            # time_iter = time.time()
            # _, self.velocity_accel, self.pos_accel = self.sensor.accel_int()
            # self.quaternion_gyro = self.sensor.gyro_int()
            # self.ang_vel = self.sensor.gyro()
            # quaternion_vel = deriv_quat(self.ang_vel, self.quaternion_gyro)
            # self.pos_gps, self.vel_gps = self.sensor.gps()
            # self.quaternion_triad, _ = self.sensor.triad()
            # self.time_total_sens.append(time.time() - time_iter)
            
            # #SENSOR CONTROL
            # pos_vel = np.array([self.pos_accel[0], self.velocity_accel[0],
            #                     self.pos_accel[1], self.velocity_accel[1],
            #                     self.pos_accel[2], self.velocity_accel[2]])

            # if self.REAL_CTRL:
            #     self.network_in = self.aux_dl.dl_input(states, [action])
            # else:
            #     states_sens = [np.concatenate((pos_vel, self.quaternion_gyro, quaternion_vel))]                
            #     self.network_in = self.aux_dl.dl_input(states_sens, [action])
            
            pos = self.env.state[0:5:2]
            ang = self.env.ang
            for i, w_i in enumerate(self.env.w):
                self.a[i] += (w_i*time_int_step )*180/np.pi/10
    
        ang_deg = (ang[2]*180/np.pi, ang[0]*180/np.pi, ang[1]*180/np.pi)
        pos = (pos[], 0+pos[1], 5+pos[2])

       
        
        # self.quad_model.setHpr((0, 0, 0))
        # self.quad_model.setPos((0, 0, 5))
        self.quad_model.setPos(*pos)
        self.quad_model.setHpr(*ang_deg)
        for prop, a in zip(self.prop_models, self.a):
            prop.setHpr(a, 0, 0)

        #print(self.env.state[0:5:2])
        #print(self.env.mat_rot)
        return task.cont
    
