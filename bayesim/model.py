import bayesim.pmf as pmf
import pandas as pd
import deepdish as dd
from copy import deepcopy
import numpy as np

class model(object):
    """
    Todo:
        animating visualization during run
        figure out how best to feed back and run additional sims after
            subdivision
        allow for multiple types of model output
        figure out data formatting for model and observations
    """

    def __init__(self,params,ec,output_var):
        """
        Initialize with a uniform PMF over the fitting parameters.

        Args:
            fit_params (:obj:`param_list`): param_list object containing parameters to be fit and associated metadata
            ec (:obj:`list` of :obj:`str`): names of experimental conditions
            output_var (`str`): name of experimental output measurements

        Todo:
            eventually, allow multiple types of output
        """
        # should we do sanity checks here or at runtime?
        # (i.e. checking that model_func doesn't require any inputs other than
        # EC's or params already enumerated)
        #self.fit_params = argv['fit_params']
        self.fit_params = params.fit_params
        #self.model_func = argv['model']
        #self.ec = argv['ec']
        self.ec = ec
        self.output_var = output_var

        # construct initial (uniform) probability distribution
        self.probs = pmf.Pmf(self.fit_params)


    def attach_observations(self,argv):
        """
        Attach measured dataset.
        """
        mode = argv.setdefault('mode','file')

        if mode == 'file':
          self.obs_data = dd.io.load(argv['fpath'])['data']
        else:
          self.obs_data = eval(argv['name']+'()')


    def attach_model(self,argv):
        """
        Attach the model for the data, either by feeding in a file of precomputed data or a function that does the computing.argv

        Args:
            mode (`str`): either 'file' or 'function' - should only use the latter if using an analytical model
            func_name (callable): if mode='function', provide function here
            fpath (`str`): if mode='file', provide path to file

        Todo:
            Figure out best way to check for correct formatting in an input file
        """

        mode = argv.setdefault('mode','file')

        if mode == 'file':
            self.model_data = dd.io.load(argv['fpath'])
            # do some checks on formatting (and that all the observed conditions are present) and compute self.start_indices and self.end_indices...
        else:
            model_func = argv['func_name']
            # iterate over parameter space and measured conditions to compute output at every point
            param_vecs = {p['name']:[] for p in self.fit_params}
            ec_vecs = {c:[] for c in self.ec}
            model_vals = []
            # self.start_indices and end_indices are indexed the same way as self.prob.points and will be a quick way to get to the model data for a given point in parameter space and then only search through the different experimental conditions
            self.start_indices = []
            self.end_indices = []
            for pt in self.probs.points.iterrows():
                param_vals = {p['name']:pt[1][p['name']] for p in self.fit_params}
                self.start_indices.append(len(model_vals))
                for d in self.obs_data.iterrows():
                    ec_vals = {c:d[1][c] for c in self.ec}
                    # add param and EC vals to the columns
                    for p in self.fit_params:
                        param_vecs[p['name']].append(param_vals[p['name']])
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

    #def attach_probs(self, pmf):
        # to put in a PMF from before

    def get_model_data(self,ec,params):
        """
        Look up modeled data from self.model_data DataFrame.

        Both args should be dicts.

        This is almost certainly not implemented in the most efficient way currently.
        """
        # find index in self.probs of these param values
        p_query_str = ''
        param_names = [p['name'] for p in self.fit_params]
        for p in param_names:
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

    def run(self):
        """
        Do Bayes!
        Will stop iterating through observations if/when 2/3 of probability mass is concentrated in <= 1/10 of boxes and decide it's time to subdivide. (completely arbitrary thresholding for now)
        """
        # randomize observation order first
        self.obs_data = self.obs_data.sample(frac=1)
        for obs in self.obs_data.iterrows():
            # hacky error approximation for now
            err = max(0.2*abs(obs[1][self.output_var]),0.01)
            lkl = self.probs.likelihood(obs[1], obs[1][self.output_var], err, self.get_model_data)
            self.probs.multiply(lkl)
            if np.sum(self.probs.most_probable(int(0.1*len(self.probs.points)))['prob'])>0.67:
                print('time to subdivide!')
                break

        #return self.probs

        # maybe return something? or just write to a file?
