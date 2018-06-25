import bayesim.pmf as pmf
import pandas as pd
import deepdish as dd

class model(object):
    """
    docstring
    """

    def __init__(self,argv):
        """
        Initialize with a uniform PMF over the fitting parameters.

        Args:
            fit_params (:obj:`param_list`): param_list object containing parameters to be fit and associated metadata
            ec (:obj:`list` of :obj:`str`): names of experimental conditions
        """
        # should we do sanity checks here or at runtime?
        # (i.e. checking that model_func doesn't require any inputs other than
        # EC's or params already enumerated)
        self.fit_params = argv['fit_params']
        #self.model_func = argv['model']
        self.ec = argv['ec']

        # construct initial (uniform) probability distribution
        self.probs = pmf.Pmf(self.fit_params)


    def attach_observations(self,argv):
        """
        Attach measured dataset.
        """
        mode = argv.setdefault('mode','file')

        if mode == 'file':
          self.obs_data = dd.io.load(argv.setdefault('name','data')+ '.hdf5')
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
            # do some checks on formatting and compute self.start_indices and self.end_indices...
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
                    model_vals.append(model_func(param_vals,ec_vals))
                self.end_indices.append(len(model_vals))
            # merge dictionaries together to put into a model data df
            vecs = deepcopy(param_vecs)
            vecs.update(ec_vecs)
            vecs.update({'model_val':model_vals})

            self.model_data = pd.DataFrame.from_dict(vecs)

    #def attach_probs(self, pmf):
        # to put in a PMF from before


    def run(self):
        # do Bayes
        # randomize observation order first?
        for obs in self.obs_data:
            lkl = self.probs.likelihood(obs)
            self.probs.multiply(lkl)

        return self.probs

        # maybe return something? or just write to a file?
