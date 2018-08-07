import numpy as np
import sys
import math
import pandas as pd
sys.path.append('../../')
import bayesim.model as bym
import bayesim.params as byp
import deepdish as dd
import matplotlib.pyplot as plt

# First, define our model
def model_y(ec, params):
    """
    Simple model of a kinematic trajectory.

    Args:
        ec (`dict`): dictionary with a key 't' leading to a float value
        params (`dict`): dictionary with keys 'v0' and 'g' leading to float values

    Returns:
        float: y-value at the given time assuming the given parameters
    """
    # read in the inputs
    t = ec['t']
    v0 = params['v0']
    g = params['g']

    # compute and return
    return v0*t - 0.5 * g * t**2

# describe our parameters
pl = byp.Param_list()
pl.add_fit_param(name='v0', val_range=[0,20], units='m/s')
pl.add_fit_param(name='g', val_range=[0,20], units='m/s^2')
pl.add_ec(name='t', units='s', is_x=True) # plot this on the x-axis
pl.add_output(name='y', units='m')

# the first two "observations"
data = pd.DataFrame()
data['t'] = [2, 2.3]
data['y'] = [3, 0.1]
data['uncertainty'] = [0.2, 0.5]
dd.io.save('two_points.h5', data)

# initialize bayesim model object
m = bym.Model(params=pl, obs_data_path='two_points.h5', model_data_func=model_y, calc_model_unc=True)

m.run()

m.visualize_probs(fpath='two_obs_probs.png')

# now let's add some more "observed" data by just generating it using our model function and some parameter values consistent with what we've done so far
data = pd.DataFrame()
t_vals = np.arange(0,3,0.1)
y_vals = [model_y({'t':t},{'v0':11.31,'g':9.81}) for t in t_vals]
data['t'] = t_vals
data['y'] = y_vals
dd.io.save('obs_data.h5',data)

# initialize bayesim model object again, now with more data (and assuming the larger uncertainty value for all the points)
m = bym.Model(params=pl, obs_data_path='obs_data.h5', model_data_func=model_y, calc_model_unc=True, fixed_unc=0.5)

# run, using all data points
m.run(min_num_pts=len(m.obs_data))

m.visualize_probs(fpath='probs_1.png')
m.comparison_plot(fpath='comp_1.png')

# now subdivide, do further "simulations", and run inference again
m.subdivide()

new_pts = dd.io.load('new_sim_points_1.h5')
new_sims = []
for pt in new_pts.iterrows():
    t = pt[1]['t']
    params = pt[1][m.fit_param_names()]
    y = model_y({'t':t}, params)
    this_pt = [t, y] + [pt[1][n] for n in m.fit_param_names()]
    new_sims.append(this_pt)
columns = ['t', 'y'] + [n for n in m.fit_param_names()]
new_sim_data = pd.DataFrame.from_records(data=new_sims, columns=columns)
dd.io.save('new_sim_data_1.h5', new_sim_data)

m.attach_model(mode='file', model_data_path='new_sim_data_1.h5', calc_model_unc=True)

m.run(min_num_pts=len(m.obs_data))

m.visualize_probs(fpath='probs_2.png')
m.comparison_plot(fpath='comp_2.png', num_param_pts=2)
