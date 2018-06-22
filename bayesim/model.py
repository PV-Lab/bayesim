import bayesim.pmf as pmf
import deepdish as dd

class model(object):
    """
    docstring
    """

    def __init__(self,argv):
        # does there need to be anything here?
        # some placeholders maybe?
        self.probs = pmf(...)

        # should we do sanity checks here or at runtime?
        # (i.e. checking that model_func doesn't require any inputs other than
        # EC's or params already enumerated)
        self.fit_params = argv['fit_params']
        #self.model_func = argv['model']
        self.ec = argv['ec']


    def attach_observations(self,argv):

        mode = argv.setdefault('mode','file')

        if mode == 'file':
          self.obs_data = dd.io.load(argv.setdefault('name','data')+ '.hdf5')
        else:
          self.obs_data = eval(argv['name']+'()')


    def attach_model(self,argv):

        mode = argv.setdefault('mode','file')

        if mode == 'file':
          self.model_data = dd.io.load(argv.setdefault('name','data')+ '.hdf5')
        else:
          loop over fit param
          self.model_data = eval(argv['name']...)


    #def attach_probs(self, pmf):
        # to put in a PMF from before


    def run(self):
        # construct PMF if necessary
     #   if self.probs==None:
     #       self.probs = pmf(...)

        # need a place to store the simulated data
        #self.sim_data = [] #DataFrame?

        # run simulations
        #for prob in self.probs:
        #    self.sim_data.append(...)

        # do Bayes
        # randomize observation order first?
        for obs in self.obs_data:
            lkl = self.probs.likelihood(obs)
            self.probs.multiply(lkl)

        return self.probs



        # maybe return something? or just write to a file?
