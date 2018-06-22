from bayesim.model import *
import math


def func_model(argv):

 n = argv['n']
 J0 = argv['J0']
 eta = 0.02585
 V = argv['V']
 T = argv['T']

 return J0*(math.exp(V/eta/T/n) -1)


def func_obs():


data = [{'V':1,'T':3,'J':0.2},\
        {'V':1,'T':4,'J':0.3}...]


return data

# also need to specify spacing (linear/log/etc.)

#fit_param = {'n':[1,2,10],'J0':[0.1,100,10]}
from bayesim import param_list
p = param_list()
p.add_fit_param(name='n',val_range=[1,2],length=10,spacing='linear')
p.add_fit_param(name='J0',val_range=[0.1,100],length=10,spacing='log',units='A/cm^2')

ec = ['V','T']

m = model(fit_params = p,\
          ec = ec)

m.attach_observations(mode = 'function',name = func_obs)
m.attach_model(mode = 'function',name = func_model)

output =  m.run(plot=True)


n = output['n']
J0 = output['J0']
