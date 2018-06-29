from bayesim.pmf import Pmf
import bayesim.param_list as pl
import pandas as pd
import deepdish as dd
from copy import deepcopy
import numpy as np

class model(object):
    """
    Updates from 180626 meeting:
        top priorities: accepting files as input, saving state (everything else later)

        make subdivide, visualize methods here
        m= bayesim(load_state=True,filename='')
        create mocked data (diode) for model and obs in hdf5 format (so GR can use it for the command line version)
        add option to run() for how often to save intermediate PMFs
        maybe also a boolean for whether it's waiting for new model data
        create new internal variable new_sims_list or something
        also add a method to return new_sims_list either as df or write to file, etc.
        fix inputs to different functions (messy)
        make sure everything can accept files as inputs (important for argparse)
        add save_state function that writes hdf5 with all current variables
        maybe make entire state into a dict?

    Todo:
        animating visualization during run
        figure out how best to feed back and run additional sims after
            subdivision
        allow for multiple types of model output
        figure out data formatting for model and observations
        put some of the pmf functions here to call directly (visualize, subdivide, etc.)
        add data plotting options
        possibly start and end indices should be stored directly in probs to avoid accidentally sorting or deleting from one and not the other
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
            self.ec = state['ec']
            self.output_var = state['output_var']
            self.ec_tol_digits = state['ec_tol_digits']

            # probabilities
            self.probs = Pmf(self.fit_params)
            self.probs.points = state['probs_points']
            self.num_sub = state['num_sub']

            # model
            self.model_data = state['model_data']
            self.needs_new_model_data = state['needs_new_model_data']
            self.obs_data = state['obs_data']
            self.start_indices = state['start_indices']
            self.end_indices = state['end_indices']
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
                self.ec = argv['ec']
            else:
                self.ec = []

            # placeholders
            self.model_data = []
            self.needs_new_model_data = True
            self.obs_data = []
            self.start_indices = []
            self.end_indices = []
            self.is_run = False
            self.num_sub = 0 # number of subdivisions done


    def attach_ec(self,ec_list):
        """
        Attach names of experimental conditions.

        Args:
            ec_list (:obj:`list` of :obj:`str`): names of experimental conditions
        """
        self.ec = ec_list

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
            self.probs = Pmf(params=params)

        else: # it's just a list of names
            self.param_names = params

    def attach_observations(self,**argv):
        """
        Attach measured dataset.

        Args:
            output_column (`str`): optional, header of column containing output data (required if different from self.output_var)

        Todo:
            generate and save list of all sets of EC's that were measured
            add option to just feed in a DataFrame
        """
        mode = argv.setdefault('mode','file')
        output_col = argv.setdefault('output_column',self.output_var)

        if mode == 'file':
            self.obs_data = dd.io.load(argv['fpath'])
            # get EC names if necessary
            cols = list(self.obs_data.columns)
            if output_col not in cols:
                raise NameError('Your output variable name, %s, is not the name of a column in your input data!' %(output_col))
                return
            else:
                cols.remove(output_col)
                if self.ec == []:
                    print('Identified experimental conditions as %s. If this is wrong, rerun and explicitly specify them with attach_ec (make sure they match data file columns) or remove extra columns from data file.' %(str(cols)))
                    self.ec = cols
                else:
                    if set(cols) == set(self.ec):
                        pass # all good
                    elif set(ec) <= set(cols):
                        print('Ignoring extra columns in data file: %s'%(str(list(set(cols)-set(ec)))))
                    elif set(cols) <= set(ec):
                        print('These experimental conditions were missing from your data file: %s\nProceeding assuming that %s is the full set of experimental conditions...'%(str(list(set(ec)-set(cols))), str(cols)))
                        self.ec = cols

        else:
            # this option hasn't been tested and is maybe unnecessary
            self.obs_data = eval(argv['name']+'()')

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
            for c in self.ec:
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
        ec_pts = pd.DataFrame.from_records(data=self.obs_data.groupby(self.ec).groups.keys(),columns=self.ec).round(self.ec_tol_digits).sort_values(ec).reset_index(drop=True)
        # then check at each model point that they match
        for name,group in grps:
            if not all(ec_pts==group.round(self.ec_tol_digits).sort_values(ec).reset_index(drop=True)):
                raise ValueError('The experimental conditions do not match to %d digits between the observed and modeled data at the modeled parameter space point %s!'%(self.ec_tol_digits,name))
                return

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
            some way to elegantly use the function when adding new model data?
        """

        #mode = argv.setdefault('mode','file')
        mode = argv['mode']
        output_col = argv.setdefault('output_column',self.output_var)

        if mode == 'file':
            # import and sort data on parameter values
            self.model_data = dd.io.load(argv['fpath']).sort_values(self.param_names)

            # Check that columns match EC's and parameter names
            self.check_data_columns(model_data=self.model_data,output_column=output_col)

            # next get list of parameter space points
            param_points_grps = self.model_data.groupby(self.param_names)
            param_points = pd.DataFrame.from_records(data=param_points_grps.groups.keys(),columns=param_names).sort_values(self.param_names).reset_index(drop=True)

            # if PMF has been populated, check that points match
            if not self.probs == [] and not all(param_points==probs.points[self.param_names]):
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

            # compute self.start_indices and self.end_indices...
            # (rewrite using groupby and compare speeds)
            ind = 0
            for pt in self.probs.points.iterrows():
                param_vals = {p:pt[1][p] for p in self.param_names}
                query_str = ''
                for n in param_vals.keys():
                    query_str = query_str + 'abs(%f-%s)/%s<1e-6 & '%(param_vals[n],n,n)
                query_str = query_str[:-3]

                subset = self.model_data.query(query_str)
                start_ind = subset.index[0]
                end_ind = subset.index[-1]+1
                self.start_indices.append(start_ind)
                self.end_indices.append(end_ind)
        elif mode=='add':
            # import and sort data
            new_data = dd.io.load(argv['fpath']).sort_values(self.param_names)

            # check that columns are okay
            self.check_data_columns(model_data=new_data,output_column=output_col)

            # next get list of parameter space points
            new_points_grps = self.model_data.groupby(self.param_names)
            new_points = pd.DataFrame.from_records(data=param_points_grps.groups.keys(),columns=param_names).sort_values(self.param_names).reset_index(drop=True)

            # check that the points are the right ones
            if not new_points == self.probs.points[self.param_names].tail(len(new_points)).sort_values(self.param_names).reset_index(drop=True):
                # probably shouldn't rely on ordering but rather explicitly save new_pts somewhere or something like that
                raise ValueError('The parameter points in your newly added data do not match the newly added points in the PMF!')
                return

            # check that all the EC's are there
            self.check_ecs(gb=new_points_grps)

            # append the model data

            # calculate the new start and end indices

        elif mode=='function':
            # is there a way to save this (that could be saved to HDF5 too) so that subdivide can automatically call it?
            model_func = argv['func_name']
            # iterate over parameter space and measured conditions to compute output at every point
            param_vecs = {p:[] for p in self.param_names}
            ec_vecs = {c:[] for c in self.ec}
            model_vals = []
            # self.start_indices and end_indices are indexed the same way as self.prob.points and will be a quick way to get to the model data for a given point in parameter space and then only search through the different experimental conditions
            for pt in self.probs.points.iterrows():
                param_vals = {p:pt[1][p] for p in self.param_names}
                self.start_indices.append(len(model_vals))
                for d in self.obs_data.iterrows():
                    ec_vals = {c:d[1][c] for c in self.ec}
                    # add param and EC vals to the columns
                    for p in self.param_names:
                        param_vecs[p].append(param_vals[p])
                    for c in self.ec:
                        ec_vecs[c].append(d[1][c])
                    # compute/look up the model data
                    # need to make sure that model_func takes in params and EC in appropriate format
                    model_vals.append(model_func(ec_vals,param_vals))
                self.end_indices.append(len(model_vals))
            # merge dictionaries together to put into a model data df
            vecs = deepcopy(param_vecs)
            vecs.update(ec_vecs)
            vecs.update({self.output_var:model_vals})

            self.model_data = pd.DataFrame.from_dict(vecs)

        self.needs_new_model_data = False

    #def attach_probs(self, pmf):
        # to put in a PMF from before
        # maybe not necessary with ability to load state

    def get_model_data(self,ec,params):
        """
        Look up modeled data from self.model_data DataFrame.

        Both args should be dicts.

        This is almost certainly not implemented in the most efficient way currently. Also parameter values had better not be 0.
        """
        # find index in self.probs of these param values
        p_query_str = ''
        for p in self.param_names:
            #print params, p, params[p]
            p_query_str = p_query_str + 'abs(%f-%s)/%s<1e-6 & '%(params[p],p,p)
        p_query_str = p_query_str[:-3]
        pt = self.probs.points.query(p_query_str)
        if not len(pt)==1:
            print("Something went wrong finding the modeled data! Couldn't find data at %s"%p_query_str)
        else:
            i = pt.index[0]
            ind_low = self.start_indices[i]
            ind_hi = self.end_indices[i]

        # pull out modeled data at that parameter space point...
        data_subset = self.model_data.iloc[ind_low:ind_hi]

        # now query for the EC's in question
        ec_query_str = ''
        for c in self.ec:
            ec_query_str = ec_query_str + 'abs(%f-%s)<1e-6 & '%(ec[c],c)
        ec_query_str = ec_query_str[:-3]
        pt = data_subset.query(ec_query_str)
        if not len(pt)==1:
            print("Something went wrong finding the modeled data! Couldn't find data at %s, %s"%(p_query_str,ec_query_str))
        else:
            return float(pt[self.output_var])

    def run(self, **argv):
        """
        Do Bayes!
        Will stop iterating through observations if/when 2/3 of probability mass is concentrated in <= 1/10 of boxes and decide it's time to subdivide. (completely arbitrary thresholding for now)

        Args:
            save_step (`int`): interval (number of data points) at which to save intermediate PMF's (defaults to 10, 0 to save only final, <0 to save none)
            th_pm (`float`): threshold quantity of probability mass to be concentrated in th_pv fraction of parameter space to trigger the run to stop (defaults to 0.8)
            th_pv (`float`): threshold fraction of parameter space volume for th_pm fraction of probability to be concentrated into to trigger the run to stop (defaults to 0.05)
        """
        save_step = argv.setdefault('save_step',10)
        th_pm = argv.setdefault('th_pm',0.8)
        th_pv = argv.setdefault('th_pv',0.05)
        # need to test whether I need to specifically parse into the namespace if they're different from defaults

        # randomize observation order first
        self.obs_data = self.obs_data.sample(frac=1)
        count = 0
        for obs in self.obs_data.iterrows():
            # hacky error approximation for now
            err = max(0.2*abs(obs[1][self.output_var]),0.01)
            lkl = self.probs.likelihood(obs[1], obs[1][self.output_var], err, self.get_model_data)
            self.probs.multiply(lkl)
            if save_step >0 and count % save_step == 0:
                dd.io.save('PMF_%d.h5'%(count),self.probs.points)
            if np.sum(self.probs.most_probable(int(0.05*len(self.probs.points)))['prob'])>0.8:
                print('time to subdivide!')
                if save_step >=0:
                    dd.io.save('PMF_final.h5',self.probs.points)
                self.is_run = True
                break
            else:
                count = count + 1

    def subdivide(self, **argv):
        """
        Subdivide the probability distribution and save the list of new sims to run to a file.

        Args:
            threshold_prob (`float`): minimum probability of box to subdivide (default 0.001)

        """
        threshold_prob = argv.setdefault('threshold_prob',0.001)
        self.num_sub = self.num_sub + 1
        filename = 'new_sim_points_%d.h5'%(self.num_sub)
        new_boxes, dropped_inds = self.probs.subdivide(threshold_prob)
        # remove start and end indices for subdivided boxes
        for i in sorted(dropped_inds,reverse=True):
            del self.start_indices[i]
            del self.end_indices[i]
        self.needs_new_model_data = True
        self.is_run = False
        dd.io.save(filename,new_boxes)
        print('New model points to simulate are saved in the file %s.'%filename)

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
        state['ec'] = self.ec
        state['ec_tol_digits'] = self.ec_tol_digits
        state['output_var'] = self.output_var

        # PMF
        state['probs_points'] = self.probs.points
        state['num_sub'] = self.num_sub

        # model/data
        state['model_data'] = self.model_data
        state['needs_new_model_data'] = self.needs_new_model_data
        state['obs_data'] = self.obs_data
        state['start_indices'] = self.start_indices
        state['end_indices'] = self.end_indices
        state['is_run'] = self.is_run

        # save the file
        dd.io.save(filename,state)
