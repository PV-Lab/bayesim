from bayesim.pmf import Pmf
import bayesim.param_list as pl
import pandas as pd
import deepdish as dd
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import sys

class model(object):
    """
    The main workhorse class of bayesim. Stores the modeled and observed data as well as a Pmf object which maintains the current probability distribution and grid subdivisions.

    Attributes:
    [update this]
    """

    def __init__(self,**argv):
        """
        Initialize with a uniform PMF over the fitting parameters.

        Args:
            fit_params (:obj:`param_list`): param_list object containing parameters to be fit and associated metadata
            ec (:obj:`list` of :obj:`str`): names of experimental conditions
            ec_tol_digits(`int`): number of digits to round off values of EC's (default 5)
            output_var (`str`): name of experimental output measurements
            load_state (`bool`): flag for whether to load state from a file - if True, other inputs (apart from state_file) are ignored
            state_file (`str`): path to file saved by save_state() fcn
        """

        state_file = argv.get('state_file','bayesim_state.h5')
        ec_tol_digits = argv.get('ec_tol_digits',5)
        load_state = argv.get('load_state',False)

        if load_state:
            state = dd.io.load(state_file)

            # variables
            self.fit_params = state['fit_params']
            self.param_names = [p['name'] for p in self.fit_params]
            self.ec_names = state['ec']
            self.ec_pts = state['ec_pts']
            self.output_var = state['output_var']
            self.ec_tol_digits = state['ec_tol_digits']
            self.ec_x_var = state['ec_x_var']

            # probabilities
            self.probs = Pmf(params=self.fit_params)
            self.probs.points = state['probs_points']
            self.probs.num_sub = state['num_sub']

            # model
            self.model_data = state['model_data']
            self.model_data_grps = state['model_data_grps']
            self.model_data_ecgrps = state['model_data_ecgrps']
            self.needs_new_model_data = state['needs_new_model_data']
            self.obs_data = state['obs_data']
            self.is_run = state['is_run']


        else:
            # read in inputs
            self.output_var = argv['output_var']
            self.ec_tol_digits = ec_tol_digits

            if 'params' in argv.keys():
                self.attach_params(argv['params'])
            else:
                self.fit_params = []
                self.param_names = []

            if 'ec' in argv.keys():
                self.ec_names = argv['ec']
            else:
                self.ec_names = []

            # placeholders
            self.probs = Pmf()
            self.ec_pts = pd.DataFrame()
            self.model_data = pd.DataFrame()
            self.model_data_grps = []
            self.model_data_ecgrps = []
            self.needs_new_model_data = True
            self.obs_data = pd.DataFrame()
            self.is_run = False
            self.ec_x_var = ''

    def attach_ec_names(self,ec_list):
        """
        Attach names of experimental conditions.

        Args:
            ec_list (:obj:`list` of :obj:`str`): names of experimental conditions
        """
        self.ec_names = ec_list

    def attach_params(self,params):
        """
        Attach list of parameters to fit.

        Args:
            param_list: list of strings OR param_list object
        """

        if isinstance(params, pl.param_list):
            # then we can make the PMF now
            self.fit_params = params.fit_params
            self.param_names = [p['name'] for p in self.fit_params]
            self.probs = Pmf(params=params.fit_params)

        else: # it's just a list of names
            self.param_names = params

    def attach_observations(self, **argv):
        """
        Attach measured dataset.

        Args:
            fpath (`str`): path to HDF5 file containing observed data
            keep_all (`bool`): whether to keep all the data in the file (longer simulation times) or to clip out data points that are close to each other (defaults to False)
            ec_x_var (`str`): required if keep_all is False, the experimental condition over which to measure differences (e.g. V for JV(Ti) curves in PV). It will also be used in plotting later.
            max_ec_x_step (`float`): used if keep_all is False, largest step to take in the ec_x_var before keeping a point even if curve if "flat" (defaults to 0.05 * range of ec_x_var)
            thresh_dif_frac (`float`): used if keep_all is False, threshold (as a percentage of the maximum value, defaults to 0.03)
            fixed_error (`float`): required if running in function mode or if file doesn't have an 'error' column, value to use as uncertainty in measurement
            output_column (`str`): optional, header of column containing output data (required if different from self.output_var)
        """
        output_col = argv.get('output_column', self.output_var)
        keep_all = argv.get('keep_all', True)
        thresh_dif_frac = argv.get('thresh_dif_frac', 0.01)
        if 'ec_x_var' in argv.keys():
            self.ec_x_var = argv['ec_x_var']
        elif not keep_all:
            raise NameError('You must specify ec_x_var if you want to throw out data points that are too close together.')

        self.obs_data = dd.io.load(argv['fpath'])
        # get EC names if necessary
        cols = list(self.obs_data.columns)
        if output_col not in cols:
            raise NameError('Your output variable name, %s, is not the name of a column in your input data!' %(output_col))
            return
        elif 'fixed_error' not in argv.keys() and 'error' not in cols:
            raise NameError('You need to either provide a value for fixed_error or your measurement data needs to have an error column!')
            return
        else:
            cols.remove(output_col)
            if self.ec_names == []:
                self.ec_names = [c for c in cols if not c=='error']
                print('Identified experimental conditions as %s. If this is wrong, rerun and explicitly specify them with attach_ec (make sure they match data file columns) or remove extra columns from data file.' %(str(self.ec_names)))
            else:
                if 'error' not in cols:
                    self.obs_data['error'] = argv['fixed_error']*np.ones(len(self.obs_data))
                    cols.extend('error')
                if set(cols) == set(self.ec_names+['error']):
                    pass # all good
                elif set(self.ec_names+['error']) <= set(cols):
                    print('Ignoring extra columns in data file: %s'%(str(list(set(cols)-set(ec)))))
                elif set(cols) <= set(self.ec_names+['error']):
                    print('These experimental conditions were missing from your data file: %s\nProceeding assuming that %s is the full set of experimental conditions...'%(str(list(set(ec)-set(cols))), str(cols)))
                    self.ec_names = cols

        # pick out rows to keep - the way to do this thresholding should probably be tweaked
        if not keep_all:
            other_ecs = [ec for ec in self.ec_names if not ec==self.ec_x_var]
            obs_data_grps = self.obs_data.groupby(by=other_ecs)
            for grp in obs_data_grps.groups.keys():
                subset = deepcopy(self.obs_data.loc[obs_data_grps.groups[grp]]).sort_values(self.ec_x_var)
                if 'max_ec_x_step' in argv.keys():
                    max_step = argv['max_ec_x_step']
                else:
                    max_step = 0.1 * max(subset[self.ec_x_var]-min(subset[self.ec_x_var]))
                    print('Using %.2f as the maximum step size in %s when choosing observation points to keep.'%(max_step,self.ec_x_var))
                thresh = thresh_dif_frac * (max(subset[self.output_var])-min(subset[self.output_var]))
                i = 0
                while i < len(subset)-1:
                    this_pt = subset.iloc[i]
                    next_pt = subset.iloc[i+1]
                    if next_pt[self.ec_x_var]-this_pt[self.ec_x_var] >= max_step:
                        i = i+1
                    elif next_pt[self.output_var]-this_pt[self.output_var] < thresh:
                        subset.drop(next_pt.name,inplace=True)
                        self.obs_data.drop(next_pt.name,inplace=True)
                    else:
                        i = i+1

        # round EC values
        rd_dct = {n:self.ec_tol_digits for n in self.ec_names}
        self.obs_data = self.obs_data.round(rd_dct)

        # sort observed data
        self.obs_data.sort_values(by=self.ec_names)

        # populate list of EC points
        self.ec_pts =  pd.DataFrame.from_records(data=[list(k) for k in self.obs_data.groupby(self.ec_names).groups.keys()],columns=self.ec_names).round(self.ec_tol_digits).sort_values(self.ec_names).reset_index(drop=True)

    def check_data_columns(self,**argv):
        """
        Make sure the columns in imported data make sense.

        Args:
            model_data (`DataFrame`): dataset to check
            output_column (`str`): optional, header of column containing output data (required if different from self.output_var)
        """
        model_data = argv['model_data']
        output_col = argv['output_column']

        cols = list(model_data.columns)
        # first, check that the output is there
        if output_col not in cols:
            raise NameError('Your output variable name, %s, is not the name of a column in your model data!' %(output_col))
            return
        else:
            cols.remove(output_col)
            # next, that all the experimental conditions are there
            for c in self.ec_names:
                if c not in cols:
                    raise NameError('Experimental condition %s is not the name of a column in your model data!' %(c))
                    return
                else:
                    cols.remove(c)
            # if param_names has been populated, check if they match
            if not self.param_names == []:
                if set(self.param_names)==set(cols):
                    pass # all good
                elif set(self.param_names) <= set(cols):
                    print('Ignoring extra columns in model data file: %s'%(str(list(set(cols)-set(self.param_names)))))
                elif set(cols) <= set(self.param_names):
                    print('These experimental conditions were missing from your model data file: %s\nProceeding assuming that %s is the full set of experimental conditions...'%(str(list(set(self.param_names)-set(cols))), str(cols)))
                    self.param_names = cols
            # if param_names wasn't populated, populate it
            else:
                self.param_names = cols

    def check_ecs(self,**argv):
        """
        Check that all experimental conditions are present at each parameter point in modeled data.

        Args:
            gb (:obj:`groupby`): Pandas groupby object of model data grouped by parameter points
        """
        grps = argv['gb']
        # then check at each model point that they match
        for name,group in grps:
            if not all(self.ec_pts==group[self.ec_names].round(self.ec_tol_digits).sort_values(self.ec_names).reset_index(drop=True)):
                raise ValueError('The experimental conditions do not match to %d digits between the observed and modeled data at the modeled parameter space point %s!'%(self.ec_tol_digits,name))
                return

    def calc_indices(self):
        """
        Compute starting and ending indices in self.model_data for each point in self.probs.
        """
        start_indices = np.zeros(len(self.probs.points),dtype=int)
        end_indices = np.zeros(len(self.probs.points),dtype=int)
        self.model_data_grps = self.model_data.groupby(by=self.param_names)
        for pt in self.probs.points.iterrows():
            subset_inds = self.model_data_grps.groups[tuple(pt[1][self.param_names].tolist())]
            if len(subset_inds)==0:
                print('Something went wrong calculating sim indices! Could not find any points in model data for params %s.'%pt)
            start_ind = int(min(subset_inds))
            end_ind = int(max(subset_inds))
            start_indices[pt[0]] = start_ind
            end_indices[pt[0]] = end_ind
        self.probs.points['start_ind'] = start_indices
        self.probs.points['end_ind'] = end_indices

    def attach_model(self,**argv):
        """
        Attach the model for the data, either by feeding in a file of precomputed data or a function that does the computing.

        Args:
            mode (`str`): 'file' or 'function'
            func_name (callable): if mode='function', provide function here
            fpath (`str`): if mode=='file', provide path to file
            output_column (`str`): optional, header of column containing output data (required if different from self.output_var)
        """

        mode = argv['mode']
        output_col = argv.get('output_column',self.output_var)

        if mode == 'file':
            # import and sort data on parameter values
            self.model_data = dd.io.load(argv['fpath']).sort_values(self.param_names+self.ec_names)

            # Check that columns match EC's and parameter names
            self.check_data_columns(model_data=self.model_data,output_column=output_col)

            # next get list of parameter space points
            param_points_grps = self.model_data.groupby(self.param_names)
            param_points = pd.DataFrame.from_records(data=[list(k) for k in param_points_grps.groups.keys()],columns=self.param_names).sort_values(self.param_names).reset_index(drop=True)

            # if PMF has been populated, check that points match
            if not self.probs == [] and not all(param_points==self.probs.points[self.param_names]):
                print('Your previously populated PMF does not have the same set of parameter space points as your model data. Proceeding using the points from the model data.')
                self.probs = []

            ## check that all EC's are present at all model points
            # first get list of EC points from observed data (round off and sort values before comparison)
            self.check_ecs(gb=param_points_grps)

            # Generate self.probs if necessary
            if self.probs == []:
                # check that points are on a grid (the quick but slightly less certain way)
                param_lengths = [len(set(param_points[name])) for name in self.param_names]
                if not np.product(param_lengths)==len(param_points):
                    raise ValueError('Your modeled parameter space does not appear to be on a grid; the current version of bayesim can only handle initially gridded spaces (unless using a previously saved subdivided state).')
                    return
                else:
                    param_vals = {name:list(set(param_points[name])) for name in self.param_names}
                    # try to guess spacing - this may need twiddling
                    param_spacing = {}
                    for name in self.param_names:
                        vals = param_vals[name]
                        diffs = [vals[i+1]-vals[i] for i in range(len(vals)-1)]
                        ratios = [vals[i+1]/vals[i] for i in range(len(vals)-1)]
                        if np.std(diffs)/np.mean(diffs) < np.std(ratios)/np.mean(ratios):
                            param_spacing[name] = 'linear'
                        else:
                            param_spacing[name] = 'log'

                    params = pl.param_list()
                    for name,vals in param_vals:
                        params.add_fit_param(name=name,vals=vals)

            # generate self.params if necessary? (might be done by here)

        elif mode=='function':
            # is there a way to save this (that could be saved to HDF5 too) so that subdivide can automatically call it?
            model_func = argv['func_name']
            # iterate over parameter space and measured conditions to compute output at every point
            param_vecs = {p:[] for p in self.param_names}
            ec_vecs = {c:[] for c in self.ec_names}
            model_vals = []

            for pt in self.probs.points.iterrows():
                param_vals = {p:pt[1][p] for p in self.param_names}
                for d in self.obs_data.iterrows():
                    ec_vals = {c:d[1][c] for c in self.ec_names}
                    # add param and EC vals to the columns
                    for p in self.param_names:
                        param_vecs[p].append(param_vals[p])
                    for c in self.ec_names:
                        ec_vecs[c].append(d[1][c])
                    # compute/look up the model data
                    # need to make sure that model_func takes in params and EC in appropriate format
                    model_vals.append(model_func(ec_vals,param_vals))

            # merge dictionaries together to put into a model data df
            vecs = deepcopy(param_vecs)
            vecs.update(ec_vecs)
            vecs.update({self.output_var:model_vals})

            self.model_data = pd.DataFrame.from_dict(vecs)

        # reset index to avoid duplication
        self.model_data.reset_index(inplace=True,drop=True)

        # round EC's and generate groups
        rd_dct = {n:self.ec_tol_digits for n in self.ec_names}
        self.model_data = self.model_data.round(rd_dct)
        self.model_data_ecgrps = self.model_data.groupby(self.ec_names)

        # update flag and indices
        self.needs_new_model_data = False
        self.calc_indices()

    def comparison_plot(self,**argv):
        """
        Plot observed data vs. highest-probability modeled data.

        Args:
            ecs (`dict`): optional, dict of EC values at which to plot. If not provided, they will be chosen randomly. This can also be a list of dicts for multiple points.
            num_ecs (`int`): number of EC values to plot, defaults to 1 (ignored if ecs is provided)
            num_param_pts (`int`): number of the most probable parameter space points to plot (defaults to 1)
            ec_x_var (`str`): one of self.ec_names, will overwrite if this was provided before in attach_observations, required if it wasn't. If ec was provided, this will supercede that
            fpath (`str`): optional, path to save image to if desired (if num_plots>1, this will be used as a prefix)
        """
        # read in options and do some sanity checks
        num_ecs = argv.get('num_ecs',1)
        if 'ecs' in argv.keys():
            ecs = argv['ecs']
            if not (isinstance(ecs,list) or isinstance(ecs,np.ndarray)):
                ecs = [ecs]
        else:
            ecs = []
            for i in range(num_ecs):
                ec_tuple = random.choice(list(self.model_data_ecgrps.groups.keys()))
                ecs.append({self.ec_names[i]:ec_tuple[i] for i in range(len(ec_tuple))})

        num_param_pts = argv.get('num_param_pts',1)

        if 'ec_x_var' in argv.keys():
            self.ec_x_var = argv['ec_x_var']

        if not hasattr(self,'ec_x_var'):
            print('You have not provided an x-variable from your experimental conditions against which to plot. Choosing the first one in the list, %s.'%(self.ec_names[0]))
            self.ec_x_var = self.ec_names[0]

        # need to fix this check
        #if self.ec_x_var in ecs.keys():
            #print('You provided a fixed value for your x variable, ignoring that and plotting the full range.')
            #del ecs[self.ec_x_var]

        other_ecs = [n for n in self.ec_names if not n==self.ec_x_var]

        param_pts = self.probs.most_probable(num_param_pts)

        fig, axs = plt.subplots(len(ecs),sharex=True,figsize=(6,4*len(ecs)))

        for i in range(len(ecs)):
            if len(ecs)>1:
                ax = axs[i]
            else:
                ax = axs
            ax.set_prop_cycle(None)
            ec = ecs[i]
            obs_data = self.obs_data
            plot_title = ''
            for c in other_ecs:
                obs_data =  obs_data[abs(obs_data[c]-ec[c])<=10.**(-1.*self.ec_tol_digits)]
                plot_title = plot_title + '%s=%f, '%(c,ec[c])
            obs_data = obs_data.sort_values(by=[self.ec_x_var])
            ax.plot(obs_data[self.ec_x_var],obs_data[self.output_var])
            j = 1
            legend_list = ['observed']
            for pt in param_pts.iterrows():
                model_data = self.model_data.loc[self.model_data_grps.groups[tuple([pt[1][n] for n in self.param_names])]]
                for c in other_ecs:
                    model_data =  model_data[abs(model_data[c]-ec[c])<=10.**(-1.*self.ec_tol_digits)]
                model_data.sort_values(by=[self.ec_x_var])
                ax.plot(model_data[self.ec_x_var],model_data[self.output_var])
                leg_label = 'modeled: '
                for p in self.param_names:
                    leg_label = leg_label + '%s=%f, '%(p,pt[1][p])
                leg_label = leg_label[:-2]
                legend_list.append(leg_label)
                j = j + 1

            # set ylims to match observed data
            obs_max = max(obs_data[self.output_var])
            obs_min = min(obs_data[self.output_var])
            obs_width = obs_max-obs_min
            ax.set_ylim([obs_min-0.05*obs_width,obs_max+0.05*obs_width])
            plt.xlabel(self.ec_x_var)
            ax.set_ylabel(self.output_var)
            ax.legend(legend_list)
            plot_title = plot_title[:-2]
            ax.set_title(plot_title)

    def run(self, **argv):
        """
        Do Bayes!
        Will stop iterating through observations if/when >= th_pm of probability mass is concentrated in <= th_pv of boxes and decide it's time to subdivide. (completely arbitrary thresholding for now)

        Args:
            save_step (`int`): interval (number of data points) at which to save intermediate PMF's (defaults to 10, 0 to save only final, <0 to save none)
            th_pm (`float`): threshold quantity of probability mass to be concentrated in th_pv fraction of parameter space to trigger the run to stop (defaults to 0.8)
            th_pv (`float`): threshold fraction of parameter space volume for th_pm fraction of probability to be concentrated into to trigger the run to stop (defaults to 0.05)
            min_num_pts (`int`): minimum number of observation points to use - if threshold is reached before this number of points has been used, it will start over and the final PMF will be the average of the number of runs needed to use sufficient points (defaults to 50)
            force_exp_err (`bool`): If true, likelihood calculations will use only experimental errors and ignore the computed model errors.
        """
        # read in options
        save_step = argv.get('save_step',10)
        th_pm = argv.get('th_pm',0.8)
        th_pv = argv.get('th_pv',0.05)
        min_num_pts = argv.get('min_num_pts',50)
        force_exp_err = argv.get('force_exp_err',False)

        if min_num_pts > len(self.obs_data):
            print('Can not use more observation points than there are in the data. Setting min_num_pts to len(self.obs_data)=%d'%len(self.obs_data))
            min_num_pts = len(self.obs_data)

        # do some sanity checks
        if self.needs_new_model_data:
            raise ValueError('You need to attach model data before running!')
            return

        if self.is_run:
            print('Running again at the same subdivision level. Previous results may be overridden...')

        # set up folders for intermediate files
        if save_step>0:
            folder = os.getcwd()
            pmf_folder = folder+'/PMFs/'
            obs_list_folder = folder+'/obs_lists/'
            for fp in [pmf_folder,obs_list_folder]:
                if not os.path.isdir(fp):
                    os.mkdir(fp)

        # set error = deltas (might want to tweak this)
        self.model_data['error'] = self.model_data['deltas']

        # TESTING THIS
        # rather than uniformizing, averaging the previous probs with a quasi-uniform to see if it fixes weird sampling things
        old_probs = deepcopy(self.probs)
        uni_probs = deepcopy(self.probs)
        uni_probs.uniformize()
        mid_probs = deepcopy(self.probs)
        mid_probs.points['prob'] = 0.9*old_probs.points['prob'] + 0.1*uni_probs.points['prob']
        mid_probs.normalize()


        # randomize observation order
        self.obs_data = self.obs_data.sample(frac=1)
        num_pts_used = 0
        num_runs = 0
        probs_lists = []
        delta_count_list = []
        while num_pts_used < min_num_pts:
            prev_used_pts = num_pts_used
            num_runs = num_runs + 1
            obs_indices = []

            #self.probs.uniformize()
            # TESTING
            self.probs = mid_probs

            done=False
            while not done:
                # check if we've gone through all the points
                if num_pts_used == len(self.obs_data):
                    print('Used all the observed data! Last PMF to go into average may have been further from threshold.')
                    done = True
                else:
                    # get observed and modeled data
                    obs = self.obs_data.iloc[num_pts_used]
                    obs_indices.append(num_pts_used)
                    ec = obs[self.ec_names]
                    ecpt = tuple([ec[n] for n in self.ec_names])
                    model_here = deepcopy(self.model_data.loc[self.model_data_ecgrps.groups[ecpt]])

                    # compute likelihood and do a Bayesian update
                    lkl, delta_count = self.probs.likelihood(meas=obs, model_at_ec=model_here,output_col=self.output_var,force_exp_err=force_exp_err)
                    self.probs.multiply(lkl)
                    num_pts_used = num_pts_used + 1
                    delta_count_list.append(delta_count)

                    # save intermediate PMF if necessary
                    if save_step >0 and (num_pts_used-prev_used_pts) % save_step == 0:
                        dd.io.save(pmf_folder+'sub%d_run%d_PMF_%d.h5'%(self.probs.num_sub,num_runs,num_pts_used-prev_used_pts),self.probs.points)

                    # check if threshold probability concentration has been reached
                    if np.sum(self.probs.most_probable(int(th_pv*len(self.probs.points)))['prob'])>th_pm:
                        done = True

                if done:
                    probs_lists.append(np.array(self.probs.points['prob']))
                    if save_step >= 0:
                        dd.io.save(pmf_folder+'sub%d_run%d_PMF_final.h5'%(self.probs.num_sub,num_runs),self.probs.points)
                    at_threshold=True
                    dd.io.save(obs_list_folder+'sub%d_run%d_obs_list.h5'%(self.probs.num_sub,num_runs),self.obs_data.iloc[obs_indices])

        probs = np.mean(probs_lists,axis=0)
        self.probs.points['prob'] = probs
        print('Did a total of %d runs to use a total of %d observations.'%(num_runs,num_pts_used))

        print('\nAn average of %d / %d probability points used model errors (rather than experimental errors) during this run.'%(int(round(np.mean(delta_count_list))),len(self.probs.points)))

        if save_step >=0:
            dd.io.save(pmf_folder+'sub%d_PMF_final.h5'%(self.probs.num_sub),self.probs.points)
        self.is_run = True

    def subdivide(self, **argv):
        """
        Subdivide the probability distribution and save the list of new sims to run to a file.

        Args:
            threshold_prob (`float`): minimum probability of box to (keep and) subdivide (default 0.001)
        """
        threshold_prob = argv.get('threshold_prob',0.001)
        new_boxes = self.probs.subdivide(threshold_prob)
        #dropped_inds = list(dropped_boxes.index)
        self.fit_params = [p for p in self.probs.params]

        # remove old model data
        self.model_data = pd.DataFrame()

        # update flags
        self.needs_new_model_data = True
        self.is_run = False
        #dd.io.save(filename,new_boxes)
        filename = 'new_sim_points_%d.h5'%(self.probs.num_sub)
        self.list_model_pts_to_run(fpath=filename)
        print('New model points to simulate are saved in the file %s.'%filename)

    def list_model_pts_to_run(self,fpath):
        """
        Generate full list of model points that need to be run (not just parameter points but also all experimental conditions). Saves to HDF5 at fpath.

        Note that this could be very slow if used on the initial grid (i.e. for potentially millions of points) - it's better for after a subdivide call.
        """

        # get just the columns with the parameters
        param_pts = self.probs.points[self.param_names]

        # Now at every point in that list, make a row for every EC point
        param_inds = range(len(param_pts))
        ec_inds = range(len(self.ec_pts))
        columns = self.param_names + self.ec_names
        pts = []
        for ppt in param_pts.iterrows():
            for ecpt in self.ec_pts.iterrows():
                pts.append(list(ppt[1])+list(ecpt[1]))
        sim_pts = pd.DataFrame(data=pts,columns=columns)
        dd.io.save(fpath,sim_pts)

    def calc_model_gradients(self):
        """
        Calculates largest difference in modeled output along any parameter direction for each experimental condition, to be used for error in calculating likelihoods. Currently only works if data is on a grid.

        (also assumes it's sorted by param names and then EC's)
        """
        param_lengths = [p['length'] for p in self.fit_params]

        deltas = np.zeros(len(self.model_data))

        # for every set of conditions...
        for grp in self.model_data_ecgrps.groups:
            inds = self.model_data_ecgrps.groups[grp]
            # construct matrix of output_var({fit_params})
            subset = deepcopy(self.model_data.loc[inds])
            # sort and reset index of subset to match probs so we can use the find_neighbor_boxes function if needed
            subset.drop_duplicates(subset=self.param_names,inplace=True)
            subset.sort_values(self.param_names,inplace=True)
            subset.reset_index(inplace=True)
            if not all(subset[self.param_names]==self.probs.points[self.param_names]):
                raise ValueError('Subset at %s does not match probability grid!'%grp)

            # check if on a grid
            if not len(subset)==np.product(param_lengths):
                is_grid = False
                # construct grid at the highest level of subdivision
                dense_grid = self.probs.populate_dense_grid(df=subset,col_to_pull=self.output_var,make_ind_lists=True)
                mat = dense_grid['mat']
                ind_lists = dense_grid['ind_lists']

            else:
                is_grid = True
                mat = np.reshape(list(subset[self.output_var]), param_lengths)

            # given matrix, compute largest differences along any direction
            winner_dim = [len(mat.shape)]
            winner_dim.extend(mat.shape)
            winners = np.zeros(winner_dim)

            for i in range(len(mat.shape)):
                # build delta matrix
                deltas_here = np.absolute(np.diff(mat,axis=i))
                pad_widths = [(0,0) for j in range(len(mat.shape))]
                pad_widths[i] = (1,1)
                deltas_here = np.pad(deltas_here, pad_widths, mode='constant', constant_values=0)

                # build "winner" matrix (ignore nans)
                winners[i]=np.fmax(deltas_here[[Ellipsis]+[slice(None,mat.shape[i],None)]+[slice(None)]*(len(mat.shape)-i-1)],deltas_here[[Ellipsis]+[slice(1,mat.shape[i]+1,None)]+[slice(None)]*(len(mat.shape)-i-1)])

                grad = np.amax(winners,axis=0)

            # save these values to the appropriate indices in the vector
            if is_grid:
                deltas[inds] = grad.flatten()
            else:
                # pick out only the boxes that exist
                deltas[inds] = grad[[i for i in list([ind_lists[p] for p in self.param_names])]]
            self.model_data['deltas'] = deltas

            #count = count+1

    def save_state(self,filename='bayesim_state.h5'):
        """
        Save the entire state of this model object to an HDF5 file so that work can be resumed later.
        """

        # construct a dict with all the state variables
        state = {}

        # parameters
        state['fit_params'] = self.fit_params
        state['ec'] = self.ec_names
        state['ec_pts'] = self.ec_pts
        state['ec_tol_digits'] = self.ec_tol_digits
        state['ec_x_var'] = self.ec_x_var
        state['output_var'] = self.output_var

        # PMF
        state['probs_points'] = self.probs.points
        state['num_sub'] = self.probs.num_sub

        # model/data
        state['model_data'] = self.model_data
        state['model_data_grps'] = self.model_data_grps
        state['model_data_ecgrps'] = self.model_data_ecgrps
        state['needs_new_model_data'] = self.needs_new_model_data
        state['obs_data'] = self.obs_data
        state['is_run'] = self.is_run

        # save the file
        dd.io.save(filename,state)

    def visualize_grid(self,**argv):
        """
        Visualize the current state of the grid.

        Args:
            same as pmf.visualize()
        """
        self.probs.visualize(just_grid=True,**argv)

    def visualize_probs(self,**argv):
        """
        Visualize the PMF with a corner plot.

        Args:
            same as pmf.visualize()
        """
        self.probs.visualize(**argv)
