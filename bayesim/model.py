from bayesim.pmf import Pmf
import bayesim.params as pm
from bayesim.utils import calc_deltas
import pandas as pd
import deepdish as dd
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import sys
from joblib import Parallel, delayed, cpu_count
import time
import platform

class Model(object):
    """
    The main workhorse class of bayesim. Stores the modeled and observed data as well as a Pmf object which maintains the current probability distribution and grid subdivisions.

    Attributes:
    [update this]
    """

    def __init__(self,**argv):
        """
        Initialize with a uniform PMF over the fitting parameters.

        Args:
            obs_data_path (`str`): path to HDF5 file containing measured data
            load_state (`bool`): flag for whether to load state from a file - if True, other inputs (apart from state_file) are ignored
            state_file (`str`): path to file saved by save_state() fcn
            verbose (`bool`): flag for verbosity, defaults to False
            output_var (`str`): name of experimental output measurements
            params (:obj:`param_list`): Param_list object containing parameters to be fit and associated metadata
            ec_x_var (`str`): EC to plot on x-axis and consider in trimming data
            ec_list (:obj:`list` of :obj:`str`): names of experimental conditions
            ec_tols (`dict`): dict of form {ec_name_1:tolerance_1, ec_name_2:tolerance_2, ...}, will supersede ec_list
            ec_units (`dict`): dict of form {ec_name_1:units_1, ec_name_2:units_2, ...}, optional
            model_data_path (`str`): path to HDF5 file containing modeled data
            model_data_func (callable): handle to function for computing model data
        """
        verbose = argv.get('verbose', False)
        state_file = argv.get('state_file', 'bayesim_state.h5')
        load_state = argv.get('load_state', False)

        if load_state:
            if verbose:
                print('Loading bayesim state from %s...\n'%state_file)
            state = dd.io.load(state_file)

            # variables
            self.ec_pts = state['ec_pts']
            self.output_var = state['output_var']
            self.params = pm.Param_list(param_dict=state['params'])

            # probabilities
            self.probs = Pmf(prob_dict=state['probs'])

            # model
            self.model_data = state['model_data']
            self.model_data_grps = state['model_data_grps']
            self.model_data_ecgrps = state['model_data_ecgrps']
            self.needs_new_model_data = state['needs_new_model_data']
            self.obs_data = state['obs_data']
            self.is_run = state['is_run']

        else:
            if verbose:
                print('Constructing bayesim model object...\n')
            # initialize empty parameter list
            self.params = pm.Param_list()

            # if a param_list object has been provided, attach it
            if 'params' in argv.keys():
                self.attach_params(argv['params'])
            else:
                self.probs = Pmf()

            # check if output list is populated / attach output variable
            output_list = self.params.output
            if len(output_list)>0: # outputs are populated
                if len(output_list)>1 or ('output_var' in argv.keys() and not output_list[0].name==argv['output_var']):
                    raise ValueError('It seems you are trying to add more than one output variable. Sorry, the current version of bayesim supports only one type of output - this will be addressed in future versions!')
                else:
                    self.output_var = output_list[0].name # for now
            elif 'output_var' in argv.keys():
                self.params.add_output(name=argv['output_var'])
                self.output_var = argv['output_var'] # for now...eventually need to be general to multiple potential outputs
            else:
                raise ValueError("You need to define your output variable!")

            # attach EC's if provided explicitly, or infer from obs_data
            if len(set(['ec_list','ec_tols','ec_units']).intersection(set(argv.keys())))>0:
                self.attach_ecs(**argv)

            # set x-axis EC variable
            if 'ec_x_var' in argv.keys():
                self.params.set_ec_x(argv['ec_x_var'])

            # attach observed data if provided
            if 'obs_data_path' in argv.keys():
                self.attach_observations(**argv)
            else:
                self.ec_pts = pd.DataFrame()
                self.obs_data = pd.DataFrame

            # attach modeled data if provided
            if 'model_data_path' in argv.keys():
                self.attach_model(mode='file', **argv)
            elif 'model_data_func' in argv.keys():
                self.attach_model(mode='function', **argv)
            else:
                self.model_data = pd.DataFrame()
                self.model_data_grps = []
                self.model_data_ecgrps = []
                self.needs_new_model_data = True

            # run flag
            self.is_run = False

    def attach_params(self, params):
        """Attach a param_list object."""
        if not self.params.is_empty():
            print('Overwriting preexiting parameter list with this new one.\n')
        self.params = params
        self.probs = Pmf(params=params.fit_params)

    def attach_ecs(self, **argv):
        """
        Define parameters for experimental conditions.

        Args:
            ec_list (:obj:`list` of :obj:`str`): names of experimental conditions
            ec_tols (`dict`): dict of form {ec_name_1:tolerance_1, ec_name_2:tolerance_2, ...}, will supersede ec_list
            ec_units (`dict`): dict of form {ec_name_1:units_1, ec_name_2:units_2, ...}, optional
        """
        ec_names = []
        tol_dict = {}
        unit_dict = {}
        if 'ec_tols' in argv.keys():
            tol_dict = argv['ec_tols']
            ec_names = tol_dict.keys()
        if 'ec_units' in argv.keys():
            unit_dict = argv['ec_units']
            if ec_names==[]:
                ec_names = unit_dict.keys()
        if 'ec_list' in argv.keys():
            if ec_names==[]:
                ec_names = argv['ec_list']

        for name in ec_names:
            args = {'name':name}
            if name in tol_dict.keys():
                args['tolerance'] = tol_dict[name]
            if name in unit_dict.keys():
                args['units'] = unit_dict[name]
            self.params.add_ec(**args)

    def attach_fit_params(self,params):
        """
        Attach list of parameters to fit.

        Args:
            param_list: list of Fit_param objects
        """
        for param in params:
            self.params.add_fit_param(param=param)

    def attach_observations(self, **argv):
        """
        Attach measured dataset.

        Args:
            obs_data_path (`str`): path to HDF5 file containing observed data
            keep_all (`bool`): whether to keep all the data in the file (longer simulation times) or to clip out data points that are close to each other (defaults to False)
            ec_x_var (`str`): required if keep_all is False, the experimental condition over which to measure differences (e.g. V for JV(Ti) curves in PV). It will also be used in plotting later.
            max_ec_x_step (`float`): used if keep_all is False, largest step to take in the ec_x_var before keeping a point even if curve if "flat" (defaults to 0.05 * range of ec_x_var)
            thresh_dif_frac (`float`): used if keep_all is False, threshold (as a percentage of the maximum value, defaults to 0.03)
            fixed_unc (`float`): required if running in function mode or if file doesn't have an 'uncertainty' column, value to use as uncertainty in measurement
            output_column (`str`): optional, header of column containing output data (required if different from self.output_var)
            verbose (`bool`): flag for verbosity, defaults to False
        """
        output_col = argv.get('output_column', self.output_var)
        keep_all = argv.get('keep_all', True)
        thresh_dif_frac = argv.get('thresh_dif_frac', 0.01)
        verbose = argv.get('verbose', False)

        if 'ec_x_var' in argv.keys():
            self.params.set_ec_x(argv['ec_x_var'])
        elif not keep_all and self.params.ec_x_name==None:
            raise NameError('You must specify ec_x_var if you want to throw out data points that are too close together.')

        if verbose:
            print('Attaching measured data...\n')

        self.obs_data = dd.io.load(argv['obs_data_path'])
        # get EC names if necessary
        cols = list(self.obs_data.columns)
        if output_col not in cols:
            raise NameError('Your output variable name, %s, is not the name of a column in your input data!' %(output_col))
            return
        elif 'fixed_unc' not in argv.keys() and 'uncertainty' not in cols:
            raise NameError('You need to either provide a value for fixed_unc or your measurement data needs to have an uncertainty column!')
            return
        else:
            cols.remove(output_col)
            if len(self.params.ecs)==0 or ('ec_x_var' in argv.keys() and len(self.params.ecs)==1):
                print('Determining experimental conditions from observed data...\n')
                for c in cols:
                    if not c=='uncertainty':
                        # 1% of the smallest difference between values
                        tol=0.01*min(abs(np.diff(list(set(self.obs_data[c])))))
                        if c==argv['ec_x_var']:
                            self.params.set_tolerance(c, tol)
                        else:
                            self.params.add_ec(name=c, tolerance=tol)
                print('Identified experimental conditions as %s. (If this is wrong, rerun and explicitly specify them with attach_ec (make sure they match data file columns) or remove extra columns from data file.)\n' %(str(self.params.param_names('ec'))))

            # these next bits used to be under an else...
            if 'uncertainty' not in cols:
                self.obs_data['uncertainty'] = argv['fixed_unc']*np.ones(len(self.obs_data))
                cols.extend('uncertainty')
                #print(self.obs_data.head())
                self.params.set_tolerance(self.output_var, 0.01*argv['fixed_unc'])
            else: # set tolerance to 1% of minimum uncertainty
                self.params.set_tolerance(self.output_var, 0.01*min(self.obs_data['uncertainty']))

            if set(cols) == set(self.ec_names()+['uncertainty']):
                pass # all good
            elif set(self.ec_names()+['uncertainty']) <= set(cols):
                print('Ignoring extra columns in data file: %s\n'%(str(list(set(cols)-set(self.ec_names())))))
            elif set(cols) <= set(self.ec_names()+['uncertainty']):
                print('These experimental conditions were missing from your data file: %s\nProceeding assuming that %s is the full set of experimental conditions...\n'%(str(list(set(ec)-set(cols))), str(cols)))
                for c in [c for c in cols if not c=='uncertainty']:
                    self.params.add_ec(name=c)
            # ...else ended here

        # pick out rows to keep - the way to do this thresholding should probably be tweaked
        if not keep_all:
            if verbose:
                print('Choosing which measured data to keep...')
            other_ecs = [ec for ec in self.ec_names() if not ec==self.params.ec_x_name]
            obs_data_grps = self.obs_data.groupby(by=other_ecs)
            for grp in obs_data_grps.groups.keys():
                subset = deepcopy(self.obs_data.loc[obs_data_grps.groups[grp]]).sort_values(self.params.ec_x_name)
                if 'max_ec_x_step' in argv.keys():
                    max_step = argv['max_ec_x_step']
                else:
                    max_step = 0.1 * max(subset[self.params.ec_x_name]-min(subset[self.params.ec_x_name]))
                    print('Using %.2f as the maximum step size in %s when choosing observation points to keep at %s=%s.\n'%(max_step, self.params.ec_x_name, other_ecs, grp))
                thresh = thresh_dif_frac * (max(subset[self.output_var])-min(subset[self.output_var]))
                i = 0
                while i < len(subset)-1:
                    this_pt = subset.iloc[i]
                    next_pt = subset.iloc[i+1]
                    if next_pt[self.params.ec_x_name]-this_pt[self.params.ec_x_name] >= max_step:
                        i = i+1
                    elif next_pt[self.output_var]-this_pt[self.output_var] < thresh:
                        subset.drop(next_pt.name,inplace=True)
                        self.obs_data.drop(next_pt.name,inplace=True)
                    else:
                        i = i+1

        # round EC values
        rd_dct = {c.name:c.tol_digits for c in self.params.ecs}
        self.obs_data = self.obs_data.round(rd_dct)

        # sort observed data
        self.obs_data.sort_values(by=self.ec_names(), inplace=True)

        # populate list of EC points
        if len(self.params.ecs)==1: # it's finicky in this case
            self.ec_pts = pd.DataFrame()
            self.ec_pts[self.ec_names()[0]] = self.obs_data.groupby(self.ec_names()).groups.keys()
        else:
            self.ec_pts = pd.DataFrame.from_records(data=[list(k) for k in self.obs_data.groupby(self.ec_names()).groups.keys()], columns=self.ec_names())

        self.ec_pts = self.ec_pts.sort_values(self.ec_names()).reset_index(drop=True)

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
            for c in self.ec_names():
                if c not in cols:
                    raise NameError('Experimental condition %s is not the name of a column in your model data!' %(c))
                    return
                else:
                    cols.remove(c)
            # if param_names has been populated, check if they match
            if not self.fit_param_names() == []:
                if set(self.fit_param_names())==set(cols):
                    pass # all good
                elif set(self.fit_param_names()) <= set(cols):
                    print('Ignoring extra columns in model data file: %s\n'%(str(list(set(cols)-set(self.fit_param_names())))))
                elif set(cols) <= set(self.fit_param_names()):
                    print('These experimental conditions were missing from your model data file: %s\nProceeding assuming that %s is the full set of experimental conditions...\n'%(str(list(set(self.fit_param_names())-set(cols))), str(cols)))
            else:
                print("Determining fitting parameters from modeled data...\n")
                for c in cols:
                    if not c=='uncertainty':
                        vals = list(set(model_data[c]))
                        self.params.add_fit_param(name=c, vals=vals)
                print("Found fitting parameters: %s"%self.fit_param_names())

    def check_ecs(self,**argv):
        """
        Check that all experimental conditions are present at each parameter point in modeled data.

        Args:
            gb (:obj:`groupby`): Pandas groupby object of model data grouped by parameter points
            verbose (`bool`): flag for verbosity, defaults to False
        """
        verbose = argv.get('verbose', False)
        if verbose:
            print('Checking that modeled data contains all experimental conditions at every combination of fit parameters...\n')

        grps = argv['gb']
        # then check at each model point that they match
        for name,group in grps:
            #print(self.ec_pts,group[self.ec_names()].sort_values(self.ec_names()).reset_index(drop=True))
            # specifically, we check that EC pts is a SUBSET of model EC's at each parameter space point
            ec_inds = self.ec_pts.index
            if not all(self.ec_pts==group[self.ec_names()].sort_values(self.ec_names()).reset_index(drop=True).loc[ec_inds]):
                # FIX MEEEEE
                #raise ValueError('The experimental conditions do not match to %d digits between the observed and modeled data at the modeled parameter space point %s!'%(self.ec_tol_digits,name))
                print('there is a problem I need to fix the error message for!')
                return

    def calc_indices(self):
        """
        Compute starting and ending indices in self.model_data for each point in self.probs.
        """
        start_indices = np.zeros(len(self.probs.points),dtype=int)
        end_indices = np.zeros(len(self.probs.points),dtype=int)
        #print(list(self.model_data_grps.groups.keys())[:5])
        for pt in self.probs.points.iterrows():
            #print(tuple(pt[1][self.fit_param_names()].tolist()))
            subset_inds = self.model_data_grps.groups[tuple(pt[1][self.fit_param_names()].tolist())]
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
            model_data_func (callable): if mode='function', provide function here
            model_data_path (`str`): if mode=='file', provide path to file
            output_column (`str`): optional, header of column containing output data (required if different from self.output_var)
            calc_model_unc (`bool`): whether to calculate model uncertainties as well, defaults to False
            verbose (`bool`): flag for verbosity, defaults to False
        """
        mode = argv['mode']
        output_col = argv.get('output_column',self.output_var)
        verbose = argv.get('verbose', False)
        calc_model_unc = argv.get('calc_model_unc', False)

        if verbose:
            print('Attaching simulated data...\n')

        if mode == 'file':
            # import and sort data on parameter values
            self.model_data = dd.io.load(argv['model_data_path'])

            # Check that columns match EC's and parameter names
            # (and determine parameter names if need be)
            self.check_data_columns(model_data=self.model_data, output_column=output_col)

            # now sort data by param names and EC's
            self.model_data = self.model_data.sort_values(self.fit_param_names()+self.ec_names()).reset_index(drop=True)

            # next get list of parameter space points
            param_points_grps = self.model_data.groupby(self.fit_param_names())
            param_points = pd.DataFrame.from_records(data=[list(k) for k in param_points_grps.groups.keys()],columns=self.fit_param_names()).sort_values(self.fit_param_names()).reset_index(drop=True)

            # if PMF has been populated, check that points match
            if not self.probs.is_empty:
                #print(len(param_points),len(self.probs.points[self.fit_param_names()]))
                if not all(param_points==self.probs.points[self.fit_param_names()]):
                    print('Your previously populated PMF does not have the same set of parameter space points as your model data. Proceeding using the points from the model data.')
                    self.probs = Pmf()

            ## check that all EC's are present at all model points
            # first get list of EC points from observed data (round off and sort values before comparison)
            self.check_ecs(gb=param_points_grps)

            # Generate self.probs if necessary
            if self.probs.is_empty:
                if verbose:
                    print('Initializing probability distribution...\n')
                # check that points are on a grid (the quick but slightly less certain way)
                param_lengths = [len(set(param_points[name])) for name in self.fit_param_names()]
                if not np.product(param_lengths)==len(param_points):
                    raise ValueError('Your modeled parameter space does not appear to be on a grid; the current version of bayesim can only handle initially gridded spaces (unless using a previously saved subdivided state).')
                    return
                else:
                    self.probs = Pmf(params=self.params.fit_params)

        elif mode=='function':
            # is there a way to save this (that could be saved to HDF5 too) so that subdivide can automatically call it?
            model_func = argv['model_data_func']

            # check that probs.points is populated...


            # iterate over parameter space and measured conditions to compute output at every point
            param_vecs = {p:[] for p in self.fit_param_names()}
            ec_vecs = {c:[] for c in self.ec_names()}
            model_vals = []

            for pt in self.probs.points.iterrows():
                param_vals = {p:pt[1][p] for p in self.fit_param_names()}
                for d in self.obs_data.iterrows():
                    ec_vals = {c:d[1][c] for c in self.ec_names()}
                    # add param and EC vals to the columns
                    for p in self.fit_param_names():
                        param_vecs[p].append(param_vals[p])
                    for c in self.ec_names():
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

        #print('before rounding')
        #print(self.model_data.head())

        # round EC's and generate groups
        rd_dct = {c.name:c.tol_digits for c in self.params.ecs}
        self.model_data = self.model_data.round(rd_dct)

        # round fit params - actually just force them to be members of the vals lists
        print("Rounding model data...")
        for p in self.params.fit_params:
            self.model_data[p.name] = [p.get_closest_val(val) for val in self.model_data[p.name]]

        # generate groups
        self.model_data_ecgrps = self.model_data.groupby(self.ec_names())
        self.model_data_grps = self.model_data.groupby(by=self.fit_param_names())

        #print('before index calc')
        #print(self.model_data.head())

        # update flag and indices
        self.needs_new_model_data = False
        self.calc_indices()

        if calc_model_unc:
            self.calc_model_unc(**argv)

    def comparison_plot(self,**argv):
        """
        Plot observed data vs. highest-probability modeled data.

        Args:
            ec_vals (`dict`): optional, dict of EC values at which to plot. If not provided, they will be chosen randomly. This can also be a list of dicts for multiple points.
            num_ecs (`int`): number of EC values to plot, defaults to 1 (ignored if ecs is provided)
            num_param_pts (`int`): number of the most probable parameter space points to plot (defaults to 1)
            ec_x_var (`str`): one of self.ec_names, will overwrite if this was provided before in attach_observations, required if it wasn't. If ec was provided, this will supercede that
            fpath (`str`): optional, path to save image to if desired
        """
        if 'ec_x_var' in argv.keys():
            self.params.set_ec_x(argv['ec_x_var'])

        if self.params.ec_x_name==None:
            print('You have not provided an x-variable from your experimental conditions against which to plot. Choosing the first one in the list, %s.\n'%(self.ec_names()[0]))
            self.params.set_ec_x(self.ec_names()[0])

        other_ecs = [c for c in self.params.ecs if not c.name==self.params.ec_x_name]
        #print(self.params.ec_x_name, [c.name for c in other_ecs])

        # read in options and do some sanity checks
        num_ecs = argv.get('num_ecs',1)

        if len(other_ecs)==0: #only one experimental condition...
            one_ec = True
            ec_vals = [0] #placeholder, basically - this is klunky
        else: # more than one
            one_ec = False
            if 'ec_vals' in argv.keys():
                ec_vals = argv['ec_vals']
                if not (isinstance(ec_vals,list) or isinstance(ec_vals,np.ndarray)):
                    ec_vals = [ec_vals]
            else:
                ec_vals = []
                ec_pts = random.sample(list(self.model_data.groupby([c.name for c in other_ecs]).groups.keys()), num_ecs)

                for pt in ec_pts:
                    if len(other_ecs)==1: #pt will just be a float rather than a tuple
                        ec_vals.append({other_ecs[0].name:pt})
                    else:
                        ec_vals.append({other_ecs[i].name:pt[i] for i in range(len(pt))})

        num_param_pts = argv.get('num_param_pts',1)

        # need to fix this check
        #if self.params.ec_x_name in ecs.keys():
            #print('You provided a fixed value for your x variable, ignoring that and plotting the full range.')
            #del ecs[self.params.ec_x_name]

        # get parameter points for which to plot data
        param_pts = self.probs.most_probable(num_param_pts)

        # set up subplots
        fig, axs = plt.subplots(len(ec_vals), 2, figsize=(13,4*len(ec_vals)), squeeze=False)

        # get color cycle
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']

        # at each set of conditions...
        for i in range(len(ec_vals)):
            obs_data = self.obs_data
            plot_title = 'Comparison  '
            if not one_ec:
                plot_title = plot_title[:-1] + 'at '
                ecs_here = ec_vals[i]
                # get obs data for correct EC values
                #print(ecs_here, obs_data.head())
                for c in other_ecs:
                    obs_data =  obs_data[abs(obs_data[c.name]-ecs_here[c.name])<=10.**(-1.*c.tol_digits)]
                    plot_title = plot_title + '%s=%s, '%(c.name,c.get_val_str(ecs_here[c.name]))
            obs_data = obs_data.sort_values(by=[self.params.ec_x_name])
            # plot obs data
            axs[i,0].plot(obs_data[self.params.ec_x_name], obs_data[self.output_var], color=colors[0])

            legend_list = ['observed']
            c_ind = 1
            # plot modeled data and errors
            for pt in param_pts.iterrows():
                color = colors[c_ind]
                model_data = self.model_data.loc[self.model_data_grps.groups[tuple([pt[1][n] for n in self.fit_param_names()])]]
                if not one_ec:
                    for c in other_ecs:
                        model_data =  model_data[abs(model_data[c.name]-ecs_here[c.name])<=10.**(-1.*c.tol_digits)]
                model_data.sort_values(by=[self.params.ec_x_name])
                errors = np.subtract(model_data[self.output_var], obs_data[self.output_var])
                axs[i,0].plot(model_data[self.params.ec_x_name], model_data[self.output_var], color=color)
                axs[i,1].plot(model_data[self.params.ec_x_name], errors, color=color)
                leg_label = 'modeled: '
                for p in self.params.fit_params:
                    leg_label = leg_label + '%s=%s, '%(p.name, p.get_val_str(pt[1][p.name]))
                leg_label = leg_label[:-2]
                legend_list.append(leg_label)
                c_ind = c_ind+1

            # set ylims to match observed data and label axes
            obs_max = max(obs_data[self.output_var])
            obs_min = min(obs_data[self.output_var])
            obs_width = obs_max-obs_min
            axs[i,0].set_ylim([obs_min-0.05*obs_width, obs_max+0.05*obs_width])
            xvar = self.params.get_ec_x()
            axs[i,0].set_xlabel('%s [%s]' %(xvar.name, xvar.units))
            axs[i,1].set_xlabel('%s [%s]' %(xvar.name, xvar.units))
            yvar = self.params.find_param(self.output_var)
            axs[i,0].set_ylabel('%s [%s]' %(yvar.name, yvar.units))
            axs[i,1].set_ylabel('%s [%s]' %('$\Delta$'+yvar.name, yvar.units))
            axs[i,0].legend(legend_list, fontsize=14)
            plot_title = plot_title[:-2]
            axs[i,0].set_title(plot_title, fontsize=20)
            if one_ec:
                err_title = 'Errors'
            else:
                err_title = plot_title + ': errors'
            axs[i,1].set_title(err_title, fontsize=20)

            for j in [0,1]:
                for item in ([axs[i][j].xaxis.label, axs[i][j].yaxis.label] + axs[i][j].get_xticklabels() + axs[i][j].get_yticklabels()):
                    item.set_fontsize(18)

        plt.tight_layout()
        if 'fpath' in argv.keys():
            plt.savefig(argv['fpath'])

    def run(self, **argv):
        """
        Do Bayes!
        Will stop iterating through observations if/when >= th_pm of probability mass is concentrated in <= th_pv of boxes and decide it's time to subdivide. (completely arbitrary thresholding for now)

        Args:
            save_step (`int`): interval (number of data points) at which to save intermediate PMF's (defaults to 10, 0 to save only final, <0 to save none)
            th_pm (`float`): threshold quantity of probability mass to be concentrated in th_pv fraction of parameter space to trigger the run to stop (defaults to 0.9)
            th_pv (`float`): threshold fraction of parameter space volume for th_pm fraction of probability to be concentrated into to trigger the run to stop (defaults to 0.05)
            min_num_pts (`int`): minimum number of observation points to use - if threshold is reached before this number of points has been used, it will start over and the final PMF will be the average of the number of runs needed to use sufficient points (defaults to 0.7 * the number of experimental measurements)
            prob_bias (`float`): number from 0 to 0.5, fraction of PMF from previous step to mix into prior for this step (defaults to 0) - higher values will likely converge faster but possibly have larger errors, especially if min_num_pts is small
            verbose (`bool`): flag for verbosity, defaults to False
        """
        # read in options
        save_step = argv.get('save_step', 10)
        th_pm = argv.get('th_pm', 0.9)
        th_pv = argv.get('th_pv', 0.05)
        min_num_pts = argv.get('min_num_pts', int(0.7*len(self.obs_data)))
        verbose = argv.get('verbose', False)
        bias = argv.get('prob_bias', 0.0)
        if bias < 0 or bias > 0.5:
            print("Bias parameter must be between 0 and 0.5 - defaulting to 0.")
            bias = 0
        if verbose:
            print('Running inference!\n')

        if min_num_pts > len(self.obs_data):
            print('Cannot use more observation points than there are in the data. Setting min_num_pts to len(self.obs_data)=%d\n'%len(self.obs_data))
            min_num_pts = len(self.obs_data)

        # do some sanity checks
        if self.needs_new_model_data:
            raise ValueError('Oops, you need to attach model data before running!')
            return

        if self.is_run:
            print('Running again at the same subdivision level. Previous results may be overridden...\n')

        # set up folders for intermediate files
        if save_step>0:
            folder = os.getcwd()
            pmf_folder = folder+'/PMFs/'
            obs_list_folder = folder+'/obs_lists/'
            for fp in [pmf_folder,obs_list_folder]:
                if not os.path.isdir(fp):
                    os.mkdir(fp)

        # rather than uniformizing, averaging the previous probs with a quasi-uniform
        old_probs = deepcopy(self.probs)
        uni_probs = deepcopy(self.probs)
        uni_probs.uniformize()
        start_probs = deepcopy(self.probs)
        start_probs.points['prob'] = bias*old_probs.points['prob'] + (1-bias)*uni_probs.points['prob']
        start_probs.normalize()

        # randomize observation order
        self.obs_data = self.obs_data.sample(frac=1)
        num_pts_used = 0
        num_runs = 0
        probs_lists = []
        delta_count_list = []
        nan_count_list = []
        while num_pts_used < min_num_pts:
            prev_used_pts = num_pts_used
            num_runs = num_runs + 1
            obs_indices = []

            self.probs = start_probs

            done=False
            while not done:
                # check if we've gone through all the points
                if num_pts_used == len(self.obs_data):
                    print('Used all the observed data! Last PMF to go into average may have been further from threshold condition.\n')
                    done = True
                else:
                    # get observed and modeled data
                    obs = self.obs_data.iloc[num_pts_used]
                    obs_indices.append(num_pts_used)
                    ec = obs[self.ec_names()]
                    if len(self.ec_names())>1:
                        ecpt = tuple([ec[n] for n in self.ec_names()])
                    else:
                        ecpt = float(ec)
                    #print(ecpt)
                    #print(self.model_data_ecgrps.groups.keys())
                    model_here = deepcopy(self.model_data.loc[self.model_data_ecgrps.groups[ecpt]])

                    # compute likelihood and do a Bayesian update
                    lkl, delta_count, nan_count = self.probs.likelihood(meas=obs, model_at_ec=model_here,output_col=self.output_var)
                    self.probs.multiply(lkl)
                    num_pts_used = num_pts_used + 1
                    delta_count_list.append(delta_count)
                    nan_count_list.append(nan_count)

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
        print('Did a total of %d runs to use a total of %d observations.\n'%(num_runs,num_pts_used))

        print('\nAn average of %d / %d probability points had larger model uncertainty than experimental uncertainty during this run.\n'%(int(round(np.mean(delta_count_list))),len(self.probs.points)))

        print('\nAn average of %.2f / %d probability points were affected by missing/NaN simulation data.\n' %(np.mean(nan_count_list), len(self.probs.points)))

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

        #self.fit_params = [p for p in self.probs.params]
        self.attach_fit_params(self.probs.params)

        # remove old model data
        self.model_data = pd.DataFrame()

        # update flags
        self.needs_new_model_data = True
        self.is_run = False
        #dd.io.save(filename,new_boxes)
        filename = 'new_sim_points_%d.h5'%(self.probs.num_sub)
        self.list_model_pts_to_run(fpath=filename)
        print('New model points to simulate are saved in the file %s.'%filename)

    def list_model_pts_to_run(self, fpath, **argv):
        """
        Generate full list of model points that need to be run (not just parameter points but also all experimental conditions). Saves to HDF5 at fpath.

        Note that this could be very slow if used on the initial grid (i.e. for potentially millions of points) - it's better for after a subdivide call.

        Args:
            fpath (`str`): path to save the list to (HDF5)
            verbose (`bool`): flag for verbosity, defaults to False
        """
        verbose = argv.get('verbose', False)
        if verbose:
            print('Listing the sets of simulation parameters that need to be run...\n')

        # get just the columns with the parameters
        param_pts = self.probs.points[self.fit_param_names()]

        # Now at every point in that list, make a row for every EC point
        param_inds = range(len(param_pts))
        ec_inds = range(len(self.ec_pts))
        columns = self.fit_param_names() + self.ec_names()
        pts = []
        for ppt in param_pts.iterrows():
            for ecpt in self.ec_pts.iterrows():
                pts.append(list(ppt[1])+list(ecpt[1]))
        sim_pts = pd.DataFrame(data=pts,columns=columns)
        dd.io.save(fpath,sim_pts)

    def calc_model_unc(self, **argv):
        """
        Calculates largest difference in modeled output along any parameter direction for each experimental condition, to be used for uncertainty in calculating likelihoods. Currently only works if data is on a grid.

        (also assumes it's sorted by param names and then EC's)

        Args:
            verbose (`bool`): flag for verbosity, defaults to False
            model_unc_factor (`float`): multiplier on deltas to give uncertainty, defaults to 0.5 - smaller probably means faster convergence, but also higher chance to miss "hot spots"
        """

        factor = argv.get('model_unc_factor', 0.5)
        verbose = argv.get('verbose', False)
        if verbose:
            print('Calculating model uncertainty...')

        param_lengths = [p.length for p in self.params.fit_params]

        start_time = time.time()

        deltas = np.zeros(len(self.model_data))

        if platform.system()=='Windows':
            # need to do it in serial
            deltas_list = []
            for grp in self.model_data_ecgrps.groups:
                deltas_list.append(calc_deltas(grp, self.model_data_ecgrps.groups[grp], param_lengths, self.model_data, self.fit_param_names(), self.probs, self.output_var))
        else:
            # parallalize!
            deltas_list = Parallel(n_jobs=cpu_count())(delayed(calc_deltas)(grp, self.model_data_ecgrps.groups[grp], param_lengths, self.model_data, self.fit_param_names(), self.probs, self.output_var) for grp in self.model_data_ecgrps.groups)

        for entry in deltas_list:
            inds = self.model_data_ecgrps.groups[entry[0]]
            deltas[inds] = entry[1]

        # trying 0.5 instead of 1.0
        self.model_data['uncertainty'] = factor * deltas

        if verbose:
            print('Calculating model uncertainty took %.2f seconds.\n'%(time.time()-start_time))

    def save_state(self,filename='bayesim_state.h5'): #rewrite this!
        """
        Save the entire state of this model object to an HDF5 file so that work can be resumed later.
        """

        # construct a dict with all the state variables
        state = {}

        # parameters
        state['params'] = self.params.as_dict()
        state['ec_pts'] = self.ec_pts
        state['output_var'] = self.output_var

        # PMF
        state['probs'] = self.probs.as_dict()

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

# maybe I'll finish this eventually
    """
    def visualize_model_unc(self,**argv):

        #Visualize model uncertainty.

        #Args:
        #    ec_vals (`dict`): optional, dict of EC values at which to plot. If not provided, they will be chosen randomly. This can also be a list of dicts for multiple points.
        #    num_ecs (`int`): number of EC values to plot, defaults to 1 (ignored if ecs is provided)

        # read in options and do some sanity checks
        num_ecs = argv.get('num_ecs',1)
        if 'ec_vals' in argv.keys():
            ec_vals = argv['ec_vals']
            if not (isinstance(ec_vals,list) or isinstance(ec_vals,np.ndarray)):
                ec_vals = [ec_vals]
        else:
            ec_vals = []
            ec_pts_df = self.ec_pts.sample(num_ecs)
            ec_pts=[]
            for pt in ec_pts_df.iterrows():
                ec_vals.append({name:pt[1][name] for name in self.ec_names()})
                ec_pts.append(tuple(pt[1][self.ec_names()]))

        # set up subplots
        fig, axs = plt.subplots(len(ec_vals), figsize=(13,4*len(ec_vals)), squeeze=False)

        for i in range(len(ec_vals)):
            ecs_here = ec_vals[i]
            plot_title = ''
            for c in self.params.ecs:
                plot_title = plot_title + '%s=%s, '%(c.name,c.get_val_str(ecs_here[c.name]))
            subset = self.model_data.loc[self.model_data_ecgrps[tuple([ecs_here[n] for n in self.ec_names()])]]
            dense_grid = probs.populate_dense_grid(df=subset, col_to_pull='uncerainty', make_ind_lists=False)
            mat = dense_grid['mat']
            axs[i].imshow(mat)
            axs[i].set_title(plot_title)
    """

    def top_probs(self, num):
        """Return a DataFrame with the 'num' most probable points and some of the less interesting columns hidden."""
        df = self.probs.most_probable(num)
        cols = ['prob'] + [c for cs in [[p.name, p.name+'_min', p.name+'_max'] for p in self.params.fit_params] for c in cs]
        return df[cols]

    def set_param_info(self, param_name, **argv):
        """
        Set additional info for parameter param_name (any type).

        Args:
            param_name (str): name of parameter to modify
            units (str): units of parameter
            min_width (float): minimum width of parameter (only for fitting params)
            display_name (str): name to use on plots (can include TeX)
            tolerance (float): tolerance for this parameter
        """
        assert param_name in [p.name for p in self.params.all_params()], "I can't set info for a parameter (%s) that doesn't exist!"%param_name

        if 'units' in argv.keys():
            for i in range(len(self.params.fit_params)):
                if self.params.fit_params[i].name==param_name:
                    self.params.fit_params[i].units = argv['units']
            for i in range(len(self.params.ecs)):
                if self.params.ecs[i].name==param_name:
                    self.params.ecs[i].units = argv['units']
            for i in range(len(self.params.output)):
                if self.params.output[i].name==param_name:
                    self.params.output[i].units = argv['units']

        if 'min_width' in argv.keys():
            for i in range(len(self.params.fit_params)):
                if self.params.fit_params[i].name==param_name:
                    self.params.fit_params[i].min_width = argv['min_width']

        if 'display_name' in argv.keys():
            for i in range(len(self.params.fit_params)):
                if self.params.fit_params[i].name==param_name:
                    self.params.fit_params[i].display_name = argv['display_name']
            for i in range(len(self.params.ecs)):
                if self.params.ecs[i].name==param_name:
                    self.params.ecs[i].display_name = argv['display_name']
            for i in range(len(self.params.output)):
                if self.params.output[i].name==param_name:
                    self.params.output[i].display_name = argv['display_name']

        # pass these through to PMF since it doesn't always seem to happen automatically...
        if param_name in self.probs.param_names():
            for i in range(len(self.probs.params)):
                if self.probs.params[i].name==param_name:
                    if 'units' in argv.keys():
                        self.probs.params[i].units = argv['units']
                    if 'min_width' in argv.keys():
                        self.probs.params[i].min_width = argv['min_width']
                    if 'display_name' in argv.keys():
                        self.probs.params[i].display_name = argv['display_name']


    def ec_names(self):
        """Return list of experimental condition names."""
        return self.params.param_names('ec')

    def fit_param_names(self):
        """Return list of fitting parameter names."""
        return self.params.param_names('fit')
