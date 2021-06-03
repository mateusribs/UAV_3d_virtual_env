import sys
import time
import pandas as pd 

import numpy as np 

from environment.quadrotor_env import quad, sensor
from environment.quaternion_euler_utility import deriv_quat, euler_quat
from environment.quadrotor_control import Controller


T = 1
TIME_STEP = 0.01
TOTAL_STEPS = 4

    
class quad_sim():
    def __init__(self, render):
        
        self.done = False
        self.render = render
        
        self.quad_model = render.quad_model
        self.prop_models = render.prop_models
        self.a = np.zeros(4)
        
        #CONTROLLER POLICY
        self.quad_env = quad(TIME_STEP, TOTAL_STEPS, training = False, euler=0, T=T, clipped=True)
        self.sensor = sensor(self.quad_env)  
        
        self.inner_length = 10
        self.controller = Controller(total_time = 10, sample_time = 0.01, inner_length = self.inner_length)
        
        # self.PD = True

        self.x_wp = np.array([[0, 0, 0, 0, 0]]).T
        self.y_wp = np.array([[0, 0, 0, 0, 0]]).T
        self.z_wp = np.array([[2, 2.5, 3, 3.5, 4]]).T
        self.psi_wp = np.array([[0, 0, 0, 0, 0]]).T
        self.time = [0, 2, 4, 6, 8]
        self.step_controller = 0.01

        self.episode = 0
        
        #Planner Trajectory
        _, _, x_matrix = self.controller.getCoeff_snap(self.x_wp, self.time)
        _, _, y_matrix = self.controller.getCoeff_snap(self.y_wp, self.time)
        _, _, z_matrix = self.controller.getCoeff_snap(self.z_wp, self.time)
        _, _, psi_matrix = self.controller.getCoeff_accel(self.psi_wp, self.time)

        self.x_ref, self.dotx_ref, self.ddotx_ref, _, _ = self.controller.evaluate_equations_snap(self.time, self.step_controller, x_matrix)
        self.y_ref, self.doty_ref, self.ddoty_ref, _, _ = self.controller.evaluate_equations_snap(self.time, self.step_controller, y_matrix)
        self.z_ref, self.dotz_ref, self.ddotz_ref, _, _ = self.controller.evaluate_equations_snap(self.time, self.step_controller, z_matrix)
        self.psi_ref, _, _ = self.controller.evaluate_equations_accel(self.time, self.step_controller, psi_matrix)

        
        #CAMERA SETUP
        self.render.quad_model.setPos(0, 0, 0)
        self.render.quad_model.setHpr(0, 0, 0)
        self.d_angs = np.array([0, 0, 0])
        self.look_at = True
        self.last_time_press = 0

        
    def quad_reset_random(self):
     
        x0 = np.array([0, 0, 0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0])

        x_atual, _ = self.quad_env.reset(x0)
        
        ang = self.quad_env.ang
        ang_vel = self.quad_env.ang_vel

        pos_atual = np.array([[x_atual[0,0], x_atual[0,2], x_atual[0,4]]]).T
        vel_atual = np.array([[x_atual[0,1], x_atual[0,3], x_atual[0,5]]]).T
        ang_atual = np.array([[ang[0], ang[1], ang[2]]]).T
        ang_vel_atual = np.array([[ang_vel[0], ang_vel[1], ang_vel[2]]]).T
        
        print('Posição:', pos_atual.T)

        self.render_position(pos_atual, ang_atual)

        return pos_atual, vel_atual, ang_atual, ang_vel_atual
        
    def sensor_sp(self):
            _, self.velocity_accel, self.pos_accel = self.sensor.accel_int()
            self.quaternion_gyro = self.sensor.gyro_int()
            self.ang_vel = self.sensor.gyro()
            quaternion_vel = deriv_quat(self.ang_vel, self.quaternion_gyro)
            self.pos_gps, self.vel_gps = self.sensor.gps()
            self.quaternion_triad, _ = self.sensor.triad()
            pos_vel = np.array([self.pos_accel[0], self.velocity_accel[0],
                                self.pos_accel[1], self.velocity_accel[1],
                                self.pos_accel[2], self.velocity_accel[2]])
            states_sens = np.array([np.concatenate((pos_vel, self.quaternion_gyro, quaternion_vel))   ])
            return states_sens
   

    
    def render_position(self, position, orientation):

        pos = position
        ang = orientation

        # for i, w_i in enumerate(w):
        #     self.a[i] += (w_i*TIME_STEP)*180/np.pi/10
        ang_deg = (ang[2]*180/np.pi, ang[0]*180/np.pi, ang[1]*180/np.pi)
        pos = (pos[0], pos[1], pos[2])
        
        self.quad_model.setPos(*pos)
        self.quad_model.setHpr(*ang_deg)
        # self.render.dlightNP.setPos(*pos)
        # for prop, a in zip(self.prop_models, self.a):
        #     prop.setHpr(a, 0, 0)
           
    def step(self, pos_atual, vel_atual, ang_atual, ang_vel_atual, i):
        
        # CONTROL PD

        #Position Control
        # pos_ref = np.array([[self.x_ref[i], self.y_ref[i], self.z_ref[i]]]).T
        # vel_ref = np.array([[self.dotx_ref[i], self.doty_ref[i], self.dotz_ref[i]]]).T
        # accel_ref = np.array([[self.ddotx_ref[i], self.ddoty_ref[i], self.ddotz_ref[i]]]).T
        # # print(pos_ref, pos_atual)
        # psi = self.psi_ref[i]
        # # print(quat_z)
        # # T, phi_des, theta_des = controller.pos_control(pos_atual, pos_ref, vel_atual, vel_ref, accel_ref, psi, Kt)

        # T, phi_des, theta_des = self.controller.pos_control_PD(pos_atual, pos_ref, vel_atual, vel_ref, accel_ref, psi)
        
        # ang_des = np.array([[float(phi_des), float(theta_des), float(psi)]]).T

        # #Inner Loop - Attitude Control
        # for j in range(self.inner_length):

        #     # taux, tauy, tauz = controller.att_control(ang_atual, ang_des, ang_vel_atual, Ka)
        #     taux, tauy, tauz = self.controller.att_control_PD(ang_atual, ang_vel_atual, ang_des)

        #     action = np.array([float(T), taux, tauy, tauz])
        #     x, _, _ = self.quad_env.step(action)
            
        #     ang = self.quad_env.ang
        #     ang_vel = self.quad_env.ang_vel

        #     pos_atual = np.array([[x[0], x[2], x[4]]]).T
        #     vel_atual = np.array([[x[1], x[3], x[5]]]).T
        #     ang_atual = np.array([[ang[0], ang[1], ang[2]]]).T
        #     ang_vel_atual = np.array([[ang_vel[0], ang_vel[1], ang_vel[2]]]).T
             
        #     self.render_position(pos_atual, ang_atual)
        
        # print('Posição:', pos_atual.T)
        # print('Empuxo:', T)
        # print('Phi', phi_des)
        # print('Theta:', theta_des)

        ite = 20
        ang_des = np.array([[0, 0, 0]]).T

        for i in range(0, ite):

            # taux, tauy, tauz = controller.att_control(ang_atual, ang_des, ang_vel_atual, K)

            taux, tauy, tauz = self.controller.att_control_PD(ang_atual, ang_vel_atual, ang_des)

            action = np.array([9.81*1.03, taux, tauy, tauz])
            
            x, _, _ = self.quad_env.step(action)
            
            ang = self.quad_env.ang
            ang_vel = self.quad_env.ang_vel
            # q_atual = np.array([[x[6], x[7], x[8], x[9]]]).T
            # print('q atual:', q_atual.T)
            pos_atual = np.array([[x[0], x[2], x[4]]]).T
            vel_atual = np.array([[x[1], x[3], x[5]]]).T
            ang_atual = np.array([[ang[0], ang[1], ang[2]]]).T
            ang_vel_atual = np.array([[ang_vel[0], ang_vel[1], ang_vel[2]]]).T
            # print('Atitude:', ang_atual.T)
            # print('Velocidade Angular: ', ang_vel_atual.T)
            # self.render_position(pos_atual, ang_atual)
            # print(q_atual.T)

        return pos_atual, vel_atual, ang_atual, ang_vel_atual
        
        
        