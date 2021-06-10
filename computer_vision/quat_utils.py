import numpy as np
import math

def QuatProd(p, q):
    
    pv = np.array([p[0], p[1], p[2]], dtype='float32')
    ps = p[3]

    qv = np.array([q[0], q[1], q[2]], dtype='float32')
    qs = q[3]

    scalar = ps*qs - pv.T@qv
    vector = ps*qv + qs*pv - np.cross(pv, qv, axis=0)

    q_res = np.concatenate((vector, scalar), axis=0)

    return q_res

def computeAngles(q0, q1, q2, q3):

    roll = math.atan2(2*(q0*q1 + q2*q3), 1 - 2*(q1*q1 - q2*q2))
    pitch = math.asin(-2.0 * (q1*q3 - q0*q2))
    yaw = math.atan2(2*(q1*q2 + q0*q3), 1 - 2*(q2*q2 - q3*q3))

    return roll, pitch, yaw

def Quat2Rot(q):

    qv = np.array([q[0], q[1], q[2]], dtype='float32')
    qs = q[3]
    
    qvx = np.array([[0, -qv[2], qv[1]],
                    [qv[2], 0, -qv[0]], 
                    [-qv[1], qv[0], 0]], dtype='float32') 

    A_q = (qs**2 - qv.T@qv)*np.eye(3) + 2*qv@qv.T - 2*qs*qvx

    return A_q

def SkewMat(q):

    qx = q[0]
    qy = q[1]
    qz = q[2]

    omega = np.array([[0, -qz, qy],
                      [qz, 0, -qx],
                      [-qy, qx, 0]], dtype='float32')
    return omega

def DerivQuat(w, q):

    wx = w[0]
    wy = w[1]
    wz = w[2]   
    omega = np.array([[0, wz, -wy, wx],
                      [-wz, 0, wx, wy],
                      [wy, -wx, 0, wz],
                      [-wx, -wy, -wz, 0]], dtype='float32')
    dq = 0.5*omega@q

    return dq

def Euler2Quat(euler):

    cos_ph = np.cos(euler[0]*0.5)
    sin_ph = np.sin(euler[0]*0.5)

    cos_th = np.cos(euler[1]*0.5)
    sin_th = np.sin(euler[1]*0.5)

    cos_ps = np.cos(euler[2]*0.5)
    sin_ps = np.sin(euler[2]*0.5)

    q0 = cos_ph*cos_th*cos_ps + sin_ph*sin_th*sin_ps

    q1 = sin_ph*cos_th*cos_ps - cos_ph*sin_th*sin_ps

    q2 = cos_ph*sin_th*cos_ps + sin_ph*cos_th*sin_ps 

    q3 = cos_ph*cos_th*sin_ps - sin_ph*sin_th*cos_ps

    q = np.array([[q1, q2, q3, q0]]).T

    return q