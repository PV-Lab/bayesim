import matplotlib.pyplot as plt
import numpy as np
from itertools import product
%matplotlib inline


v_0 = np.linspace(0,20,500)
g = np.linspace(0,20,500)

def M_y(v_0, g, t):
    return v_0 * t - 0.5*g*t**2

def prob_y(v_0, g, t, meas_y, unc):
    return (1/np.sqrt(2*np.pi*unc**2)) * np.exp(-(M_y(v_0, g, t)-meas_y)**2/(2*unc**2))


contour_pts = np.reshape([prob_y(v_0[i], g[j], 2.0, 3.0, 0.2) for i in range(len(v_0)) for j in range(len(g))],(len(v_0),len(g)))
fig, ax = plt.subplots(1,2,figsize=(11,5))
plt.subplot(1,2,1)
plt.contourf(v_0, g, contour_pts, 100)
plt.tick_params(labelsize=20)
plt.xlabel('$g$ [m/s]',fontsize=22)
plt.ylabel('$v_0$ [m/s$^2$]',fontsize=22)
plt.subplot(1,2,2)
plt.contourf(v_0, g, contour_pts, 100)
plt.tick_params(labelsize=20)
plt.xlabel('$g$ [m/s]', fontsize=22)
plt.ylabel('$v_0$ [m/s$^2$]', fontsize=22)
plt.xlim([8.0, 11.0])
plt.ylim([9.0, 12.0])
plt.tight_layout()
plt.savefig('docs/img/probs_1.png', transparent=True)

v_0 = np.linspace(0,20,150)
g = np.linspace(0,20,150)
t_vals = np.arange(0,3.0,0.01)
pairs = [p for p in product(v_0,g)]
probs = {pair:prob_y(pair[0], pair[1], 2.0, 3.0, 0.2) for pair in pairs}

fig, ax = plt.subplots(1,2,figsize=(11,5))
plt.subplot(1,2,1)
for pair in probs.keys():
    if probs[pair]>1e-8:
        plt.plot(t_vals, [M_y(pair[0], pair[1], t) for t in t_vals], alpha=probs[pair]/20.0, color='b')
plt.xlabel('t [s]', fontsize=22)
plt.ylabel('y [m]', fontsize=22)
plt.tick_params(labelsize=20)
plt.xlim([-0.2, 3.0])
plt.subplot(1,2,2)
for pair in probs.keys():
    if probs[pair]>1e-15:
        plt.plot(t_vals, [M_y(pair[0], pair[1], t) for t in t_vals], alpha=probs[pair]/20.0, color='b')
plt.tick_params(labelsize=20)
plt.xlabel('t [s]', fontsize=22)
plt.ylabel('y [m]', fontsize=22)
plt.xlim([1.5, 2.5])
plt.ylim([1,5])
plt.tight_layout()
plt.savefig('docs/img/trajs.png', transparent=True)

g_vals = np.linspace(0,20,200)
v0_vals = []
