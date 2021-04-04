import matplotlib.pyplot as plt
import numpy as np





plt.style.use("fivethirtyeight")
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


ax.plot(time, q0r_list, 'r--', alpha=0.6, label = r'$w_{ref} (t)$', linewidth = 1.5)
ax.plot(time, q0e_list, 'r', label=r'$w_{est}(t)$', linewidth=1)
ax2.plot(time, q1r_list, 'g--', alpha=0.6, label = r'$p_{ref} (t)$', linewidth=1.5)
ax2.plot(time, q1e_list, 'g', label=r'$p_{est}(t)$', linewidth=1)
ax3.plot(time, q2r_list, 'b--', alpha=0.6, label = r'$q_{ref} (t)$', linewidth=1.5)
ax3.plot(time, q2e_list, 'b', label=r'$q_{est}(t)$', linewidth=1)
ax4.plot(time, q3r_list, 'y--', alpha=0.6, label = r'$r_{ref} (t)$', linewidth=1.5)
ax4.plot(time, q3e_list, 'y', label=r'$r_{est}(t)$', linewidth=1)
ax4.set_xlabel('Time (s)')

ax.set_title('Orientação Quatérnio')

# ax.set_ylim([-1, 1])
# ax2.set_ylim([-1, 1])
# ax3.set_ylim([-1, 1])
# ax4.set_ylim([-1, 1])
ax.legend()
ax2.legend()
ax3.legend()
ax4.legend()


plt.style.use("fivethirtyeight")
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


az.plot(time, rr_list, 'r--', alpha=0.6, label = r'$\phi_{ref} (t)$', linewidth = 1.5)
az.plot(time, re_list, 'r', label=r'$\phi_{est}(t)$', linewidth=1)
az2.plot(time, pr_list, 'g--', alpha=0.6, label = r'$\theta_{ref} (t)$', linewidth=1.5)
az2.plot(time, pe_list, 'g', label=r'$\theta_{est}(t)$', linewidth=1)
az3.plot(time, yr_list, 'b--', alpha=0.6, label = r'$\psi_{ref} (t)$', linewidth=1.5)
az3.plot(time, ye_list, 'b', label=r'$\psi_{est}(t)$', linewidth=1)
az3.set_xlabel('Sample')

az.set_title('Orientação Euler')

# az.set_ylim([-45, 45])
# az2.set_ylim([-45, 45])
# az3.set_ylim([-45, 45])
az.legend()
az2.legend()
az3.legend()




# fig2, (ay, ay2, ay3) = plt.subplots(3, 1, figsize=(10,10), sharex=True)
# xe_list, ye_list, ze_list, xr_list, yr_list, zr_list, time_pos = [], [], [], [], [], [], []

# pos = open("pos_data.txt", "r").read()
# lines_pos = pos.split('\n')

# for line in lines_pos:
#     if len(line)>1:
#         xe, ye, ze, xr, yr, zr, times_pos = line.split(' , ')
#         xe_list.append(float(xe))
#         ye_list.append(float(ye))
#         ze_list.append(float(ze))
#         xr_list.append(float(xr))
#         yr_list.append(float(yr))
#         zr_list.append(float(zr))
#         time_pos.append(float(times_pos))


# ay.plot(time_pos, xr_list, 'r--', alpha=0.6, label = r'$x_{ref} (t)$', linewidth = 1.5)
# ay.plot(time_pos, xe_list, 'r', label=r'$x_{est}(t)$', linewidth=1)
# ay2.plot(time_pos, yr_list, 'g--', alpha=0.6, label = r'$y_{ref} (t)$', linewidth=1.5)
# ay2.plot(time_pos, ye_list, 'g', label=r'$y_{est}(t)$', linewidth=1)
# ay3.plot(time_pos, zr_list, 'b--', alpha=0.6, label = r'$z_{ref} (t)$', linewidth=1.5)
# ay3.plot(time_pos, ze_list, 'b', label=r'$z_{est}(t)$', linewidth=1)
# ay3.set_xlabel('Sample')

# ay.set_title('Posição')

# # ax.set_ylim([-90, 90])
# # ax2.set_ylim([-90, 90])
# ay3.set_ylim([3, 5.2])
# ay.legend()
# ay2.legend()
# ay3.legend()

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

# ab.set_ylim([-45, 45])
# ab2.set_ylim([-45, 45])
# # ab3.set_ylim([-45, 45])
# ab.legend()
# ab2.legend()
# ab3.legend()

plt.tight_layout()
plt.show()