import bayesim.pmf as pmf

def __init__(self):
    # does there need to be anything here?
    # some placeholders maybe?
    self.probs = None

def attach_forward_model(self, model_func, fit_params, fixed_params):
    # fixed_params maybe not necessary - see note in pseudocode_usage
    # should we do sanity checks here or at runtime?
    # (i.e. checking that model_func doesn't require any inputs other than
    # EC's or params already enumerated)
    self.model_func = model_func


def attach_ec(self, ec):
    # define names of experimental conditions
    self.ec = ec


def attach_observations(self, func_obs):
    # what happens here? Does it call func_obs now...?
    self.obs_data = ...


def attach_probs(self, pmf):
    # to put in a PMF from before


def run(self):
    # construct PMF if necessary
    if self.probs==None:
        self.probs = pmf(...)

    # need a place to store the simulated data
    self.sim_data = [] #DataFrame?

    # run simulations
    for prob in self.probs:
        self.sim_data.append(...)

    # do Bayes
    # randomize observation order first?
    for obs in obs_data:
        lkl = self.probs.likelihood(obs)
        self.probs.multiply(lkl)

    # maybe return something? or just write to a file?
