from bayesim.model import *
import math


def func_model(argv):

 n = argv['n']
 J0 = argv['J0']
 eta = argv['eta']
 V = argv['V']
 T = argv['T']

 return J0*(math.exp(V/eta/T/n) -1)


def func_obs():


data = [{'V':1,'T':3,'J':0.2},\
        {'V':1,'T':4,'J':0.3}...]

return data

# somewhere we have to define the length of each fit dimension
# maybe the lists in fit_param have a third entry for length?
# also need to specify spacing (linear/log/etc.)

fit_param = {'n':[1,2],'J0':[0.1,100]}

# I'm wondering if this is even necessary - presumably any parameters
# that are fixed could just be fixed within the user-defined model_func?
# then if many fixed parameters needed to be read in from a file (such as
# a PC1D config file) the user could just provide that...
fixed_param = {'eta':0.02585}

ec = ['V','T']


m = model()

m.attach_forward_model(func_model,fit_param,fixed_param)

m.attach_ec(ec)

m.attach_observations(func_obs)

output =  m.run()


n = output['n']
J0 = output['J0']
