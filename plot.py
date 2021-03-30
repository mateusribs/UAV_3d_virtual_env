import matplotlib.pyplot as plt
import numpy as np

quat = True


if quat:
    plt.style.use("fivethirtyeight")
    fig, (ax, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10,10), sharex=True)
    q0e_list, q1e_list, q2e_list, q3e_list, q0r_list, q1r_list, q2r_list, q3r_list, time = [], [], [], [], [], [], [], [], []

    data = open("kf_data.txt", "r").read()
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

    ax.set_title('Orientação')

    # ax.set_ylim([-90, 90])
    # ax2.set_ylim([-90, 90])
    # ax3.set_ylim([-180, 180])
    ax.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()

    plt.tight_layout()
    plt.show()

else:
    plt.style.use("fivethirtyeight")
    fig, (ax, ax2, ax3) = plt.subplots(3, 1, figsize=(10,10), sharex=True)
    re_list, pe_list, ye_list, rr_list, pr_list, yr_list, time = [], [], [], [], [], [], []

    data = open("kf_data.txt", "r").read()
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


    ax.plot(time, rr_list, 'r--', alpha=0.6, label = r'$\phi_{ref} (t)$', linewidth = 1.5)
    ax.plot(time, re_list, 'r', label=r'$\phi_{est}(t)$', linewidth=1)
    ax2.plot(time, pr_list, 'g--', alpha=0.6, label = r'$\theta_{ref} (t)$', linewidth=1.5)
    ax2.plot(time, pe_list, 'g', label=r'$\theta_{est}(t)$', linewidth=1)
    ax3.plot(time, yr_list, 'b--', alpha=0.6, label = r'$\psi_{ref} (t)$', linewidth=1.5)
    ax3.plot(time, ye_list, 'b', label=r'$\psi_{est}(t)$', linewidth=1)
    ax3.set_xlabel('Time (s)')

    ax.set_title('Orientação')

    # ax.set_ylim([-90, 90])
    # ax2.set_ylim([-90, 90])
    # ax3.set_ylim([-180, 180])
    ax.legend()
    ax2.legend()
    ax3.legend()

    plt.tight_layout()
    plt.show()