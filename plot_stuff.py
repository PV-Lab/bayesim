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

## FIRST STEP PROBABILITIES

contour_pts = np.reshape([prob_y(v_0[i], g[j], 2.0, 3.0, 0.2) for i in range(len(v_0)) for j in range(len(g))],(len(v_0),len(g)))

fig, ax = plt.subplots(1,2,figsize=(11,5))
plt.subplot(1,2,1)
plt.contourf(v_0, g, contour_pts, 100)
plt.tick_params(labelsize=20)
rect=plt.Rectangle((8,9),3,3,fill=False,ec='r',linewidth=2)
plt.gca().add_patch(rect)
plt.xlabel('$g$ [m/s]',fontsize=22)
plt.ylabel('$v_0$ [m/s$^2$]',fontsize=22)
plt.xlim([0, 20])
plt.ylim([0, 20])
plt.subplot(1,2,2)
plt.contourf(v_0, g, contour_pts, 100)
rect=plt.Rectangle((8.02,9.01),2.97,2.97,fill=False,ec='r',linewidth=2)
plt.gca().add_patch(rect)
plt.tick_params(labelsize=20)
plt.xlabel('$g$ [m/s]', fontsize=22)
plt.ylabel('$v_0$ [m/s$^2$]', fontsize=22)
plt.xlim([8.0, 11.0])
plt.ylim([9.0, 12.0])
plt.tight_layout()
plt.savefig('docs/img/probs_1.png', transparent=True)

## FIRST STEP TRAJECTORIES

g_vals = np.linspace(0,20,100)
v0_start = np.linspace(0.9,2.1,60)
probs = {}
for g in g_vals:
    for v0_s in v0_start:
        v0 = v0_s+g
        probs[(v0,g)] = prob_y(v0, g, 2.0, 3.0, 0.2)

t_vals = np.arange(0,3.0,0.01)

fig, ax = plt.subplots(1,2,figsize=(11,5))
plt.subplot(1,2,1)
for pair in probs.keys():
    plt.plot(t_vals, [M_y(pair[0], pair[1], t) for t in t_vals], alpha=probs[pair]/250.0, color='b')
rect=plt.Rectangle((1.5,0),1,6,fill=False,ec='r',linewidth=2)
plt.gca().add_patch(rect)
plt.xlabel('t [s]', fontsize=22)
plt.ylabel('y [m]', fontsize=22)
plt.tick_params(labelsize=20)
plt.xlim([-0.2, 3.0])
plt.subplot(1,2,2)
for pair in probs.keys():
    plt.plot(t_vals, [M_y(pair[0], pair[1], t) for t in t_vals], alpha=probs[pair]/100.0, color='b')
rect=plt.Rectangle((1.505,0.01),0.99,5.955,fill=False,ec='r',linewidth=2)
plt.gca().add_patch(rect)
plt.tick_params(labelsize=20)
plt.xlabel('t [s]', fontsize=22)
plt.ylabel('y [m]', fontsize=22)
plt.xlim([1.5, 2.5])
plt.ylim([0,6])
plt.tight_layout()
plt.savefig('docs/img/trajs_1.png', transparent=True)

# SECOND STEP PROBABILITIES

contour_pts = np.reshape([prob_y(v_0[i], g[j], 2.0, 3.0, 0.2)*prob_y(v_0[i], g[j], 2.3, 0.1, 0.5) for i in range(len(v_0)) for j in range(len(g))],(len(v_0),len(g)))
fig, ax = plt.subplots(figsize=(7,6))
plt.contourf(v_0, g, contour_pts, 100)
plt.tick_params(labelsize=20)
plt.xlabel('$g$ [m/s]',fontsize=22)
plt.ylabel('$v_0$ [m/s$^2$]',fontsize=22)
plt.grid(False)
plt.xlim([5,14])
plt.ylim([6,16])
plt.savefig('docs/img/probs_2.png', transparent=True)

# SECOND STEP TRAJECTORIES

g_vals = np.linspace(5,15,50)
v0_start = np.linspace(0.9,2.1,60)
probs_next = {}
for g in g_vals:
    for v0_s in v0_start:
        v0 = v0_s+g
        probs_next[(v0,g)] = prob_y(v0, g, 2.0, 3.0, 0.2) * prob_y(v0, g, 2.3, 0.1, 0.5)

t_vals = np.arange(0,3.0,0.01)

fig, ax = plt.subplots(figsize=(7,6))
for pair in probs.keys():
    plt.plot(t_vals, [M_y(pair[0], pair[1], t) for t in t_vals], alpha=probs[pair]/900.0, color='b')
for pair in probs_next.keys():
    plt.plot(t_vals, [M_y(pair[0], pair[1], t) for t in t_vals], alpha=probs_next[pair]/100.0, color='r')
plt.xlabel('t [s]', fontsize=22)
plt.ylabel('y [m]', fontsize=22)
plt.tick_params(labelsize=20)
plt.xlim([-0.2, 3.0])
plt.grid(False)
plt.savefig('docs/img/trajs_2.png')
