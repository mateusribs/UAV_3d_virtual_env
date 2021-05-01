import matplotlib.pyplot as plt
import numpy as np
import math
import statistics
from sklearn.metrics import mean_squared_error
plt.rcParams.update({'font.size': 11})

fig, (ax, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10,10), sharex=True)
q0e_list, q1e_list, q2e_list, q3e_list, q0r_list, q1r_list, q2r_list, q3r_list, time = [], [], [], [], [], [], [], [], []

data = open("kfquat_data.txt", "r").read()
lines = data.split('\n')

for line in lines:
    if len(line)>1:
        q0e, q1e, q2e, q3e, q0r, q1r, q2r, q3r, times = line.split(' , ')
        q0e_list.append(float(q0e))
        q1e_list.append(float(q1e))
        q2e_list.append(float(q2e))
        q3e_list.append(float(q3e))
        q0r_list.append(float(q0r))
        q1r_list.append(float(q1r))
        q2r_list.append(float(q2r))
        q3r_list.append(float(q3r))
        time.append(float(times))


ax.plot(time, q0r_list, 'r--', label = r'$q_{s,ref} (t)$', linewidth = 1.5)
ax.plot(time, q0e_list, 'b', label=r'$q_{s,est}(t)$', linewidth=1)
ax2.plot(time, q1r_list, 'r--', label = r'$q_{x,ref} (t)$', linewidth=1.5)
ax2.plot(time, q1e_list, 'b', label=r'$q_{x,est}(t)$', linewidth=1)
ax3.plot(time, q2r_list, 'r--', label = r'$q_{y,ref} (t)$', linewidth=1.5)
ax3.plot(time, q2e_list, 'b', label=r'$q_{y,est}(t)$', linewidth=1)
ax4.plot(time, q3r_list, 'r--', label = r'$q_{z,ref} (t)$', linewidth=1.5)
ax4.plot(time, q3e_list, 'b', label=r'$q_{z,est}(t)$', linewidth=1)
ax4.set_xlabel('Tempo (s)')

ax.legend()
ax2.legend(loc=5)
ax3.legend(loc=5)
ax4.legend()

ax.set_ylabel(r'$q_{s}$')
ax2.set_ylabel(r'$q_{x}$')
ax3.set_ylabel(r'$q_{y}$')
ax4.set_ylabel(r'$q_{z}$')

ax.grid()
ax2.grid()
ax3.grid()
ax4.grid()

#-----------------------------------------------------------------------------------------------------

fig3, (az, az2, az3) = plt.subplots(3, 1, figsize=(10,10), sharex=True)
re_list, pe_list, ye_list, rr_list, pr_list, yr_list, time = [], [], [], [], [], [], []

data = open("kfeuler_data.txt", "r").read()
lines = data.split('\n')

for line in lines:
    if len(line)>1:
        re, pe, ye, rr, pr, yr, times = line.split(' , ')
        re_list.append(float(re))
        pe_list.append(float(pe))
        ye_list.append(float(ye))
        rr_list.append(float(rr))
        pr_list.append(float(pr))
        yr_list.append(float(yr))
        time.append(float(times))


az.plot(time, rr_list, 'r--', label = r'$\phi_{ref} (t)$', linewidth = 1.5)
az.plot(time, re_list, 'b', label=r'$\phi_{est}(t)$', linewidth=1)
az2.plot(time, pr_list, 'r--', label = r'$\theta_{ref} (t)$', linewidth=1.5)
az2.plot(time, pe_list, 'b', label=r'$\theta_{est}(t)$', linewidth=1)
az3.plot(time, yr_list, 'r--', label = r'$\psi_{ref} (t)$', linewidth=1.5)
az3.plot(time, ye_list, 'b', label=r'$\psi_{est}(t)$', linewidth=1)
az3.set_xlabel('Tempo(s)')

az.legend()
az2.legend()
az3.legend()

az.set_ylabel(r'$\phi(\circ)$')
az2.set_ylabel(r'$\theta(\circ)$')
az3.set_ylabel(r'$\psi(\circ)$')

az.grid()
az2.grid()
az3.grid()


fig2, (ay, ay2, ay3) = plt.subplots(3, 1, figsize=(10,10), sharex=True)
xe_list, ye_list, ze_list, xr_list, yr_list, zr_list, time_pos = [], [], [], [], [], [], []

pos = open("pos_data.txt", "r").read()
lines_pos = pos.split('\n')

for line in lines_pos:
    if len(line)>1:
        xe, ye, ze, xr, yr, zr, times_pos = line.split(' , ')
        xe_list.append(float(xe))
        ye_list.append(float(ye))
        ze_list.append(float(ze))
        xr_list.append(float(xr))
        yr_list.append(float(yr))
        zr_list.append(float(zr))
        time_pos.append(float(times_pos))


ay.plot(time_pos, xr_list, 'r--', label = r'$x_{ref} (t)$', linewidth = 1.5)
ay.plot(time_pos, xe_list, 'b', label=r'$x_{est}(t)$', linewidth=1)
ay2.plot(time_pos, yr_list, 'r--', label = r'$y_{ref} (t)$', linewidth=1.5)
ay2.plot(time_pos, ye_list, 'b', label=r'$y_{est}(t)$', linewidth=1)
ay3.plot(time_pos, zr_list, 'r--', label = r'$z_{ref} (t)$', linewidth=1.5)
ay3.plot(time_pos, ze_list, 'b', label=r'$z_{est}(t)$', linewidth=1)
ay3.set_xlabel('Tempo (s)')

ay.set_ylabel('X (m)')
ay2.set_ylabel('Y (m)')
ay3.set_ylabel('Z (m)')

ay.legend()
ay2.legend()
ay3.legend()

ay.grid()
ay2.grid()
ay3.grid()

# fig4, (ab, ab2, ab3) = plt.subplots(3, 1, figsize=(10,10), sharex=True)
# rce_list, pce_list, yce_list, rcr_list, pcr_list, ycr_list, time = [], [], [], [], [], [], []

# data = open("angle_data.txt", "r").read()
# lines = data.split('\n')

# for line in lines:
#     if len(line)>1:
#         rce, pce, yce, rcr, pcr, ycr, times = line.split(' , ')
#         rce_list.append(float(rce))
#         pce_list.append(float(pce))
#         yce_list.append(float(yce))
#         rcr_list.append(float(rcr))
#         pcr_list.append(float(pcr))
#         ycr_list.append(float(ycr))
#         time.append(float(times))


# ab.plot(time, rcr_list, 'r--', alpha=0.6, label = r'$\phi_{ref} (t)$', linewidth = 1.5)
# ab.plot(time, rce_list, 'r', label=r'$\phi_{est}(t)$', linewidth=1)
# ab2.plot(time, pcr_list, 'g--', alpha=0.6, label = r'$\theta_{ref} (t)$', linewidth=1.5)
# ab2.plot(time, pce_list, 'g', label=r'$\theta_{est}(t)$', linewidth=1)
# ab3.plot(time, ycr_list, 'b--', alpha=0.6, label = r'$\psi_{ref} (t)$', linewidth=1.5)
# ab3.plot(time, yce_list, 'b', label=r'$\psi_{est}(t)$', linewidth=1)
# ab3.set_xlabel('Sample')

# ab.set_title('Orientação Euler Camera')

# ab.legend()
# ab2.legend()
# ab3.legend()

plt.show()

#Percentual Error

# pe_x = [abs(i - j)/j for i, j in zip(xe_list, xr_list)]
# pe_y = [abs(i - j)/j for i, j in zip(ye_list, yr_list)]
# pe_z = [abs(i - j)/j for i, j in zip(ze_list, zr_list)]

# print(pe_x)
# pe_x = statistics.mean(pe_x)
# pe_y = statistics.mean(pe_y)
# pe_z = statistics.mean(pe_z)

#Mean squared error
q0_mse = mean_squared_error(q0r_list, q0e_list)
q1_mse = mean_squared_error(q1r_list, q1e_list)
q2_mse = mean_squared_error(q2r_list, q2e_list)
q3_mse = mean_squared_error(q3r_list, q3e_list)

# roll_mse = mean_squared_error(rcr_list, rce_list)
# pitch_mse = mean_squared_error(pcr_list, pce_list)
# yaw_mse = mean_squared_error(ycr_list, yce_list)

roll_cam_mse = mean_squared_error(rr_list, re_list)
pitch_cam_mse = mean_squared_error(pr_list, pe_list)
yaw_cam_mse = mean_squared_error(yr_list, ye_list)

x_mse = mean_squared_error(xr_list, xe_list)
y_mse = mean_squared_error(yr_list, ye_list)
z_mse = mean_squared_error(zr_list, ze_list)

#RMSE
q0_rmse = math.sqrt(q0_mse)
q1_rmse = math.sqrt(q1_mse)
q2_rmse = math.sqrt(q2_mse)
q3_rmse = math.sqrt(q3_mse)

# roll_rmse = math.sqrt(roll_mse)
# pitch_rmse = math.sqrt(pitch_mse)
# yaw_rmse = math.sqrt(yaw_mse)

roll_cam_rmse = math.sqrt(roll_cam_mse)
pitch_cam_rmse = math.sqrt(pitch_cam_mse)
yaw_cam_rmse = math.sqrt(yaw_cam_mse)

x_rmse = math.sqrt(x_mse)
y_rmse = math.sqrt(y_mse)
z_rmse = math.sqrt(z_mse)

print("MEKF Quaternion RMSE: \n q0 - {0} \n q1 - {1} \n q2 - {2} \n q3 - {3}".format(q0_rmse, q1_rmse, q2_rmse, q3_rmse))
print("MEKF Euler RMSE: \n Roll - {0} \n Pitch - {1} \n Yaw - {2}".format(roll_cam_rmse, pitch_cam_rmse, yaw_cam_rmse))

# print("Camera Euler RMSE: \n Roll - {0} \n Pitch - {1} \n Yaw - {2}".format(roll_cam_rmse, pitch_cam_rmse, yaw_cam_rmse))

print("Posição RMSE: \n X - {0} \n Y - {1} \n Z - {2}".format(x_rmse, y_rmse, z_rmse))

