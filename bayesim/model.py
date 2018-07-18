from bayesim.pmf import Pmf
import bayesim.param_list as pl
import pandas as pd
import deepdish as dd
from copy import deepcopy
import numpy as np

class model(object):
    """
    Updates from 180626 meeting:
        make subdivide, visualize methods here

    Todo:
        add write_query_str function to stop copying so much code for that...but test groupby speeds first
        play with hierarchical indexing to save space and/or address above point
        animating visualization during run
        figure out how best to feed back and run additional sims after
            subdivision
        allow for multiple types of model output
        figure out data formatting for model and observations
        put some of the pmf functions here to call directly (visualize, subdivide, etc.)
        add data plotting options
        possibly start and end indices should be stored directly in probs to avoid accidentally sorting or deleting from one and not the other
        maybe include EC in param_list rather than separately?
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

        Todo:
            allow feeding in model, etc. here as well
        """

        state_file = argv.setdefault('state_file','bayesim_state.h5')
        ec_tol_digits = argv.setdefault('ec_tol_digits',5)
        load_state = argv.setdefault('load_state',False)

        if load_state:
            state = dd.io.load(argv['state_file'])

            # variables
            self.fit_params = state['fit_params']
            self.param_names = [p['name'] for p in self.fit_params]
            self.ec_names = state['ec']
            self.ec_pts = state['ec_pts']
            self.output_var = state['output_var']
            self.ec_tol_digits = state['ec_tol_digits']

            # probabilities
            self.probs = Pmf(params=self.fit_params)
            self.probs.points = state['probs_points']
            self.probs.num_sub = state['num_sub']

            # model
            self.model_data = state['model_data']
            self.model_data_ecgrps = state['model_data_ecgrps']
            self.needs_new_model_data = state['needs_new_model_data']
            self.obs_data = state['obs_data']
            self.is_run = state['is_run']


        else:
            # read in inputs
            self.output_var = argv['output_var']
            self.ec_tol_digits = ec_tol_digits

            if 'params' in argv.keys():
                self.probs = []
                self.attach_params(argv['params'])
            else:
                self.fit_params = []
                self.param_names = []

            if 'ec' in argv.keys():
                self.ec_names = argv['ec']
            else:
                self.ec_names = []

            # placeholders
            self.ec_pts = []
            self.model_data = []
            self.needs_new_model_data = True
            self.obs_data = []
            self.is_run = False
            self.num_sub = 0 # number of subdivisions done


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

    def attach_observations(self,**argv):
        """
        Attach measured dataset.

        Args:
            keep_all (`bool`): whether to keep all the data in the file (longer simulation times) or to clip out data points that are close to each other (defaults to False)
            ec_x_var (`str`): required if keep_all is False, the experimental condition over which to measure differences (e.g. V for JV(Ti) curves in PV)
            max_ec_x_step (`float`): used if keep_all is False, largest step to take in the ec_x_var before keeping a point even if curve if "flat" (defaults to 0.05 * range of ec_x_var)
            thresh_dif_frac (`float`): used if keep_all is False, threshold (as a percentage of the maximum value, defaults to 0.03)
            fixed_error (`float`): required if running in function mode or if file doesn't have an 'error' column, value to use as uncertainty in measurement
            output_column (`str`): optional, header of column containing output data (required if different from self.output_var)

        Todo:
            choose points from a larger file based on percent change
            add option to just feed in a DataFrame
        """
        output_col = argv.get('output_column', self.output_var)
        keep_all = argv.get('keep_all', False)
        thresh_dif_frac = argv.get('thresh_dif_frac', 0.01)
        if not keep_all:
            ec_x_var = argv['ec_x_var']

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
                print('Identified experimental conditions as %s. If this is wrong, rerun and explicitly specify them with attach_ec (make sure they match data file columns) or remove extra columns from data file.' %(str(cols)))
                self.ec_names = cols
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
            other_ecs = [ec for ec in self.ec_names if not ec==ec_x_var]
            obs_data_grps = self.obs_data.groupby(by=other_ecs)
            for grp in obs_data_grps.groups.keys():
                subset = deepcopy(self.obs_data.loc[obs_data_grps.groups[grp]]).sort_values(ec_x_var)
                if 'max_ec_x_step' in argv.keys():
                    max_step = argv['max_ec_x_step']
                else:
                    max_step = 0.1 * max(subset[ec_x_var]-min(subset[ec_x_var]))
                    print('Using %.2f as the maximum step size in %s when choosing observation points to keep.'%(max_step,ec_x_var))
                thresh = thresh_dif_frac * (max(subset[self.output_var])-min(subset[self.output_var]))
                i = 0
                while i < len(subset)-1:
                    this_pt = subset.iloc[i]
                    next_pt = subset.iloc[i+1]
                    if next_pt[ec_x_var]-this_pt[ec_x_var] >= max_step:
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
        model_data_grps = self.model_data.groupby(by=self.param_names)
        for pt in self.probs.points.iterrows():
            subset_inds = model_data_grps.groups[tuple(pt[1][self.param_names].tolist())]
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
            mode (`str`): 'file', 'function', or 'add' - 'file' or 'function' if populating model_data for the first time, 'add' if attaching new data
            func_name (callable): if mode='function', provide function here
            fpath (`str`): if mode='file', provide path to file
            output_column (`str`): optional, header of column containing output data (required if different from self.output_var)
        Todo:
            Should adding additional model data be a separate function?
            pmf should maybe store start/end indices, either way split out computing them rather than copying code!
            when adding new data, need to calculate new deltas somehow

        """

        #mode = argv.setdefault('mode','file')
        mode = argv['mode']
        output_col = argv.setdefault('output_column',self.output_var)

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

        elif mode=='add':
            # import and sort data
            new_data = dd.io.load(argv['fpath']).sort_values(self.param_names)

            # check that columns are okay
            self.check_data_columns(model_data=new_data,output_column=output_col)

            # next get list of parameter space points
            new_points_grps = new_data.groupby(by=self.param_names)
            new_points = pd.DataFrame.from_records(data=[list(k) for k in new_points_grps.groups.keys()],columns=self.param_names).sort_values(self.param_names).reset_index(drop=True)

            # check that the points are the right ones
            ind_arr = self.probs.points['new']==True
            #if not all(new_points == self.probs.points[self.param_names].tail(len(new_points)).sort_values(self.param_names).reset_index(drop=True)):
            if not all(new_points == self.probs.points[self.param_names][ind_arr].sort_values(self.param_names).reset_index(drop=True)):
                # probably shouldn't rely on ordering but rather explicitly save new_pts somewhere or something like that
                raise ValueError('The parameter points in your newly added data do not match the newly added points in the PMF!')
                return

            # check that all the EC's are there
            self.check_ecs(gb=new_points_grps)

            # append the model data
            self.model_data = pd.concat([self.model_data,new_data])
            self.needs_new_model_data = False

            # calculate deltas?

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

    #def attach_probs(self, pmf):
        # to put in a PMF from before
        # maybe not necessary with ability to load state

    def run(self, **argv):
        """
        Do Bayes!
        Will stop iterating through observations if/when >= th_pm of probability mass is concentrated in <= th_pv of boxes and decide it's time to subdivide. (completely arbitrary thresholding for now)

        Args:
            save_step (`int`): interval (number of data points) at which to save intermediate PMF's (defaults to 10, 0 to save only final, <0 to save none)
            th_pm (`float`): threshold quantity of probability mass to be concentrated in th_pv fraction of parameter space to trigger the run to stop (defaults to 0.8)
            th_pv (`float`): threshold fraction of parameter space volume for th_pm fraction of probability to be concentrated into to trigger the run to stop (defaults to 0.05)
            num_runs (`int`): Number of times to run to threshold (final result will be average of these, defaults to 3)

        Todo:
            make default value of th_pv dimension-number-dependent
            deal with wraparound if a run goes through all the sims (unlikely)
        """
        save_step = argv.get('save_step',10)
        th_pm = argv.get('th_pm',0.8)
        th_pv = argv.get('th_pv',0.05)
        num_runs = argv.get('num_runs',3)

        if self.needs_new_model_data:
            raise ValueError('You need to attach model data before running!')
            return

        if self.is_run:
            print('Running again at the same subdivision level. Previous results may be overridden...')

        # FOR NOW set error = 2*deltas (need to tweak this probably)
        self.model_data['error'] = 2*self.model_data['deltas']

        # randomize observation order first
        self.obs_data = self.obs_data.sample(frac=1)
        i = 0
        probs_lists = []
        for j in range(num_runs):
            count = 1
            obs_indices = []
            self.probs.uniformize()
            at_threshold=False
            print('\n*****RUN %d*****'%(j+1))
            while not at_threshold:
                obs = self.obs_data.iloc[i] # use different observations for each run
                obs_indices.append(i)
                ec = obs[self.ec_names]
                ecpt = tuple([ec[n] for n in self.ec_names])
                model_here = deepcopy(self.model_data.loc[self.model_data_ecgrps.groups[ecpt]])

                lkl = self.probs.likelihood(meas=obs, model_at_ec=model_here,output_col=self.output_var)

                self.probs.multiply(lkl)
                if save_step >0 and count % save_step == 0:
                    dd.io.save('sub%d_run%d_PMF_%d.h5'%(self.num_sub,j+1,count),self.probs.points)
                if np.sum(self.probs.most_probable(int(th_pv*len(self.probs.points)))['prob'])>th_pm:
                    print('Reached threshold after %d observation points were fed in!'%(count))
                    probs_lists.append(np.array(self.probs.points['prob']))
                    if save_step >=0:
                        dd.io.save('sub%d_run%d_PMF_final.h5'%(self.num_sub,j+1),self.probs.points)
                    at_threshold=True
                    dd.io.save('sub%d_run%d_obs_list.h5'%(self.num_sub,j+1),self.obs_data.iloc[obs_indices])
                else:
                    count = count + 1
                    i = i + 1

        probs = np.mean(probs_lists,axis=0)
        self.probs.points['prob'] = probs
        if save_step >=0:
            dd.io.save('sub%d_PMF_final.h5'%(self.num_sub),self.probs.points)
        self.is_run = True

    def subdivide(self, **argv):
        """
        Subdivide the probability distribution and save the list of new sims to run to a file.

        Args:
            threshold_prob (`float`): minimum probability of box to subdivide (default 0.001)

        Todo:
            Clearing out old model data and handling deltas
        """
        threshold_prob = argv.setdefault('threshold_prob',0.001)
        self.num_sub = self.num_sub + 1
        filename = 'new_sim_points_%d.h5'%(self.num_sub)
        new_boxes, dropped_boxes = self.probs.subdivide(threshold_prob)
        #dropped_inds = list(dropped_boxes.index)

        # remove model_data for dropped boxes
        for box in dropped_boxes.iterrows():
            to_drop = self.model_data.loc[box[1]['start_ind']:box[1]['end_ind']].index
            #sprint(to_drop)
            self.model_data.drop(to_drop,inplace=True)

        # update flags
        self.needs_new_model_data = True
        self.is_run = False
        #dd.io.save(filename,new_boxes)
        self.list_model_pts_to_run(fpath=filename)
        print('New model points to simulate are saved in the file %s.'%filename)

    def list_model_pts_to_run(self,fpath):
        """
        Generate full list of model points that need to be run (not just parameter points but also all experimental conditions). Saves to HDF5 at fpath.

        Note that this could be very slow if used on the initial grid (i.e. for potentially millions of points) - it's better for after a subdivide call.

        Todo:
            make it faster somehow?
        """
        # First, find all parameter points marked as 'new' and pick out just the columns with the values
        param_pts = self.probs.points[self.probs.points['new']==True][self.param_names]

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

        Todo:
            For subdivided case, figure out how to get data on a bigger matrix to do calculation faster!!!
        """
        param_lengths = [len(set(self.probs.points[p])) for p in self.param_names]

        if 'deltas' in self.model_data.columns:
            deltas = np.array(deepcopy(self.model_data['deltas']))
        else:
            deltas = np.zeros(len(self.model_data))

        # for every set of conditions...
        #count = 0
        for grp, vals in self.model_data_ecgrps:
            #if count%5==0:print(count)
            #print(grp,vals.index)
            # construct matrix of output_var({fit_params})
            subset = deepcopy(self.model_data.loc[vals.index])
            # sort and reset index of subset to match probs so we can use the find_neighbor_boxes function if needed
            subset.drop_duplicates(subset=self.param_names,inplace=True)
            subset.sort_values(self.param_names,inplace=True)
            subset.reset_index(inplace=True)
            #print(len(subset),len(self.probs.points))
            #print(subset[self.param_names].head(),self.probs.points[self.param_names].head())
            if not all(subset[self.param_names]==self.probs.points[self.param_names]):
                raise ValueError('Subset at %s does not match probability grid!'%grp)

            # check if on a grid
            if not len(subset)==np.product(param_lengths):
                is_grid = False
                # construct grid at the highest level of subdivision
                mat_shape = []
                param_vals = {}
                pvals_indices = {}
                #indices_lists = []
                for param in self.fit_params:
                    pname = param['name']
                    min_edge = param['edges'][0]
                    max_edge = param['edges'][-1]
                    length = 2*param['length']*2**self.probs.num_sub+1
                    if param['spacing']=='linear':
                        param_vals[pname] = np.linspace(min_edge,max_edge,length)[1:-1:2]
                    elif param['spacing']=='log':
                        param_vals[pname] = np.logspace(np.log10(min_edge),np.log10(max_edge),length)[1:-1:2]
                    mat_shape.append(len(param_vals[pname]))
                    pvals_indices[pname] = {param_vals[pname][i]:i for i in range(len(param_vals[pname]))}
                    #indices_lists.append(range(len(param_vals[pname])))
                mat = np.zeros(mat_shape)
                # populate that grid...
                # at each row in subset, find list of indices into matrix that that output should go into and copy it in
                ind_lists = {p:[] for p in self.param_names}
                delta_inds_to_update = []
                for row in subset.iterrows():
                    slices = []
                    ind_tuple = []
                    param_point = self.probs.points.loc[row[0]]
                    if param_point['new']==True:
                        delta_inds_to_update.append(int(row[1]['index']))
                    p_ind = 0
                    for param in self.param_names:
                        min_val = param_point[param+'_min']
                        max_val = param_point[param+'_max']
                        inds = [pvals_indices[param][v] for v in param_vals[param] if v>min_val and v<max_val]
                        slices.append(slice(min(inds),max(inds)+1,None))
                        ind_lists[param].append(inds[0])
                        p_ind = p_ind+1
                    output_val = row[1][self.output_var]
                    mat[slices] = output_val

                # check if any zeros remain
                if np.any(abs(mat)<1e-10):
                    print('There are deltas very close to zero...watch out!')

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
                # build "winner" matrix
                winners[i]=np.maximum(deltas_here[[Ellipsis]+[slice(None,mat.shape[i],None)]+[slice(None)]*(len(mat.shape)-i-1)],deltas_here[[Ellipsis]+[slice(1,mat.shape[i]+1,None)]+[slice(None)]*(len(mat.shape)-i-1)])

                grad = np.amax(winners,axis=0)

            # save these values to the appropriate indices in the vector
            if is_grid:
                deltas[vals.index] = grad.flatten()
            else:
                # temporary array to copy all the values into
                deltas_temp = deepcopy(deltas)
                deltas_temp[vals.index] = grad[ind_lists.values()] # Numpy "advanced indexing"
                deltas[delta_inds_to_update] = deltas_temp[delta_inds_to_update]
            self.model_data['deltas'] = deltas

            #count = count+1

    def save_state(self,filename='bayesim_state.h5'):
        """
        Save the entire state of this model object to an HDF5 file so that work can be resumed later.

        Todo:
            possibly writing some potentially large things (probs_points, model_data, obs_data) to separate files and storing paths to them
        """

        # construct a dict with all the state variables
        state = {}

        # parameters
        state['fit_params'] = self.fit_params
        state['ec'] = self.ec_names
        state['ec_pts'] = self.ec_pts
        state['ec_tol_digits'] = self.ec_tol_digits
        state['output_var'] = self.output_var

        # PMF
        state['probs_points'] = self.probs.points
        state['num_sub'] = self.probs.num_sub

        # model/data
        state['model_data'] = self.model_data
        state['model_data_ecgrps'] = self.model_data_ecgrps
        state['needs_new_model_data'] = self.needs_new_model_data
        state['obs_data'] = self.obs_data
        #state['start_indices'] = self.start_indices
        #state['end_indices'] = self.end_indices
        state['is_run'] = self.is_run

        # save the file
        dd.io.save(filename,state)

    def visualize_grid(self,**argv):
        self.probs.visualize(just_grid=True,**argv)

    def visualize_probs(self,**argv):
        """
        Visualize the PMF with a corner plot.

        Args:
            same as pmf.visualize()
        """
        self.probs.visualize(**argv)
