import numpy as np
import deepdish as dd
from decimal import *
import json
from copy import deepcopy

class Param(object):
    """
    A parameter in a bayesim analysis. Can be a fitting parameter or an experimental condition.
    """

    def __init__(self, **argv):
        """
        Initialize a Param object.

        Args:
            name (`str`): name of the parameter, required
            units (`str`): units in which parameter is measured (defaults to 'unitless')
            tolerance (`float`): smallest difference between two values of this parameter to consider "real"
            display_name (str): name for plotting (can include TeX), defaults to name
        """
        self.name = argv['name']
        units = argv.get('units','unitless')
        self.units = units
        if 'tolerance' in argv.keys():
            self.set_tolerance(argv['tolerance'])
        else:
            self.tolerance = None
        if 'display_name' in argv.keys():
            self.display_name = argv['display_name']
        else:
            self.display_name = self.name

    def set_tolerance(self, tol, islog=False):
        """Set the tolerance for this parameter."""
        if hasattr(self, 'spacing'):
            if self.spacing=='log':
                self.tolerance = tol
            elif self.spacing=='linear':
                self.tol_digits = int(-1*(np.floor(np.log10(tol))))
                self.tolerance = round(tol, self.tol_digits)
        else:
            self.tol_digits = int(-1*(np.floor(np.log10(tol))))
            self.tolerance = round(tol, self.tol_digits)

    def get_val_str(self, val):
        tol_digits = int(-1*(np.floor(np.log10(self.tolerance))))
        return str(round(val, tol_digits))

class Fit_param(Param):
    """
    A bayesim fitting parameter. Because they will be initialized on a grid, each fitting parameter stores its full list of values as well as some other information such as the spacing between them and the minimum width of a box (used during grid subdivisions).
    """

    def __init__(self, **argv):
        """
        Initialize the fitting parameter.

        Args:
            name (`str`): name of the parameter, required
            units (`str`): units in which parameter is measured (defaults to 'unitless')
            tolerance (`float`): smallest difference between two values of this parameter to consider "real", defaults to 0.1 of min_width
            val_range (:obj:`list` of :obj:`float`): [min, max] (either this or vals is required)
            vals (:obj:`list` of :obj:`float`): full list of vals for this param
            length (int): initial length of this parameter (defaults to 10)
            min_width(`float`): minimum box width for this parameter - subtractive if linear spacing and divisive if logarithmic (defaults to 0.01 of total range, required if providing val_range)
            spacing (str): 'linear' or 'log' (defaults to linear)
            verbose (bool): flag for verbosity
        """
        # get spacing and length (or set defaults)
        if not 'vals' in argv.keys():
            self.spacing = argv.get('spacing','linear')
            self.length = argv.get('length',10)
        Param.__init__(self, **argv) # set name, units, tolerance
        verbose = argv.get('verbose',False)

        # sanity check
        assert ('val_range' in argv or 'vals' in argv), "You must provide a range of values for this fitting parameter!"

        # compute edges and values
        if 'vals' in argv.keys():
            vals = argv['vals']
            assert(len(vals)>=2), "vals should have at least two values"
            vals = sorted(vals)
            self.length = len(vals)
            edges = np.zeros(len(vals)+1)

            if 'spacing' in argv.keys():
                self.spacing = argv['spacing']
            else:
                # try to guess spacing
                diffs = [vals[i+1]-vals[i] for i in range(len(vals)-1)]
                ratios = [vals[i+1]/vals[i] for i in range(len(vals)-1)]
                if np.std(diffs)/np.mean(diffs) < np.std(ratios)/np.mean(ratios):
                    self.spacing = 'linear'
                else:
                    self.spacing = 'log'

            # first edge
            if self.spacing=='linear':
                edges[0] = vals[0]-0.5*(vals[1]-vals[0])
            elif self.spacing=='log':
                edges[0] = vals[0]/((vals[1]/vals[0])**0.5)
            # most of the edges
            for i in range(1,len(vals)):
                if self.spacing=='linear':
                    edges[i] = vals[i]-0.5*(vals[i]-vals[i-1])
                elif self.spacing=='log':
                    edges[i] = vals[i]/((vals[i]/vals[i-1])**0.5)
            # last edge
            if self.spacing=='linear':
                edges[len(vals)] = vals[-1]+0.5*(vals[-1]-vals[-2])
            elif self.spacing=='log':
                edges[len(vals)] = vals[-1]*((vals[-1]/vals[-2])**0.5)

            self.vals = vals
            self.val_range = [min(edges),max(edges)]
            self.edges = edges

        elif 'val_range' in argv.keys():
            self.val_range = argv['val_range']
            assert len(self.val_range)==2 and self.val_range[1]>self.val_range[0], "val_range must be of length 2 with second entry larger than first!"
            if self.spacing == 'linear':
                edges_vals = np.linspace(self.val_range[0],self.val_range[1],2*self.length+1)
            elif self.spacing == 'log':
                edges_vals = np.logspace(np.log10(self.val_range[0]),np.log10(self.val_range[1]),2*self.length+1)
            self.edges = edges_vals[::2]
            self.vals = edges_vals[1:-1:2]

        # set min_width
        if 'min_width' not in argv.keys():
            if verbose:
                print('Setting min_width automatically for %s.'%self.name)
            if self.spacing == 'linear':
                min_width = (1./(10*self.length))*(self.val_range[1]-self.val_range[0])
            elif self.spacing == 'log':
                min_width = (self.val_range[1]/self.val_range[0])**(1./(10*self.length))
        else:
            min_width = argv['min_width']
        self.min_width = min_width

        # set tolerance if not provided
        if 'tolerance' not in argv.keys():
            if self.spacing=='linear':
                self.set_tolerance(0.1*self.min_width)
            elif self.spacing=='log':
                self.set_tolerance(self.min_width**0.1, True)
        else:
            if self.spacing=='linear':
                self.set_tolerance(argv['tolerance'])
            elif self.spacing=='log':
                self.set_tolerance(argv['tolerance'], True)

        # now round all the values based on tolerance
        if self.spacing=='linear': #tolerance is subtractive
            num_digits = self.get_tol_digits()
            self.edges = [round(e, num_digits) for e in self.edges]
            self.vals = [round(v, num_digits) for v in self.vals]
            self.val_range = [round(v, num_digits) for v in self.val_range]
        elif self.spacing=='log': #tolerance is multiplicative
            for i in range(len(self.vals)):
                self.vals[i] = round(self.vals[i], self.get_tol_digits(val=self.vals[i]))
            for i in range(len(self.edges)):
                self.edges[i] = round(self.edges[i], self.get_tol_digits(val=self.edges[i]))
            for i in range(2):
                self.val_range[i] = round(self.val_range[i], self.get_tol_digits(val=self.val_range[i]))

    def get_tol_digits(self, **argv):
        """Compute number of digits to round to. 'val' must be provided if logspaced."""
        if self.spacing=='linear':
            return self.tol_digits
        elif self.spacing=='log':
            assert 'val' in argv.keys(), "If parameter is log-spaced, the value is needed to calculate tolerance digits!"
            val = argv['val']
            tol_val = abs(val - val/self.tolerance)
            return int(-1*(np.floor(np.log10(tol_val))))+1

    def get_closest_val(self, val):
        """Return closest value to val in this parameters current set of vals."""
        diffs = [abs(val-v) for v in self.vals]
        #print(diffs)
        #print(diffs, min(diffs))
        val_ind = diffs.index(min(diffs))
        closest_val = self.vals[val_ind]
        digits = self.get_tol_digits(val=closest_val)
        if abs(round(closest_val, digits-1)/round(val, digits-1)-1.0)>0.01:
            print("The values %f and %f were pretty far apart for %s..."%(val, closest_val, self.name))
        return closest_val

    def get_val_str(self, val):
        """Return a string with this parameter's value, reasonably formatted."""
        # determine if it should be in scientific notation
        if val>1e4 or val<1e-4:
            return "{:.2E}".format(Decimal(str(float(val))))
        else:
            tol_digits = self.get_tol_digits(val=val)
            if tol_digits < 1:
                return str(int(val))
            else:
                return "{0:.{1}f}".format(val, tol_digits)

class Measured_param(Param):
    """A bayesim measured parameter. Can be experimental input or output."""
    def __init__(self, **argv):
        """
        Initialize a Param object.

        Args:
            name (`str`): name of the parameter, required
            units (`str`): units in which parameter is measured (defaults to 'unitless')
            tolerance (`float`): smallest difference between two values of this parameter to consider "real," defaults to 1E-6
            param_type (`str`): 'input' or 'output', defaults to input
        """
        Param.__init__(self, **argv) # set name, units, tolerance
        self.param_type = argv.get('param_type','input')

        # set tolerance if not provided
        if 'tolerance' not in argv.keys():
            self.set_tolerance(1e-6)

class Param_list(object):
    """
    Small class to facilitate listing and comparison of bayesim parameters.
    """

    def __init__(self, **argv):
        """
        Initialize an empty Param_list, or initialize from a dict if provided.

        Args:
            param_dict (`dict`): output of a call to as_dict()
        """
        self.fit_params = []
        self.ecs = []
        self.output = []
        self.ec_x_name = None

        if 'param_dict' in argv.keys():
            param_dict = argv['param_dict']
            for fp in param_dict['fit_params']:
                self.fit_params.append(Fit_param(**fp))
            for ec in param_dict['ecs']:
                self.ecs.append(Measured_param(**ec))
            for o_var in param_dict['output']:
                self.output.append(Measured_param(**o_var))
            self.ec_x_name = param_dict['ec_x_name']

    def add_fit_param(self, **argv):
        """
        Add a fitting parameter to the list.

        Args:
            param (:class:`Fit_param`): A Fit_param object to add to the list
            name (str): name of the parameter, required if param object not passed
            units (str): units in which parameter is measured (defaults to 'unitless')
            tolerance (float): smallest difference between two values of this parameter to consider "real"
            val_range (`:obj:`list`` of :obj:`float`): [min, max] (either this or vals is required)
            vals (:obj:`list` of :obj:`float`): full list of vals for this param
            length (int): initial length of this parameter (defaults to 10)
            min_width(float): minimum box width for this parameter - subtractive if linear spacing and divisive if logarithmic (defaults to 0.01 of total range, required if providing val_range)
            spacing (str): 'linear' or 'log' (defaults to linear)
            verbose (bool): verbosity flag
        """
        if 'name' in argv.keys():
            name = argv['name']
        elif 'param' in argv.keys():
            name = argv['param'].name
        else:
            raise ValueError("You must provide either a parameter name or a Fit_param object!")
        verbose = argv.get('verbose',False)

        if not self.param_present(name):
            self.fit_params.append(Fit_param(**argv))
        else: # overwrite
            if verbose:
                print("Overwriting metadata for fitting parameter %s with new info."%name)
            param_ind = self.fit_params.index(self.find_param(name))
            if 'param' in argv.keys():
                self.fit_params[param_ind] = argv['param']
            else:
                #print(argv)
                self.fit_params[param_ind] = Fit_param(**argv)

    def add_ec(self, **argv):
        """
        Add an experimental condition.

        Args:
            name (str): name of the parameter, required
            units (str): units in which parameter is measured (defaults to 'unitless')
            tolerance (float): smallest difference between two values of this parameter to consider "real," defaults to 1E-6
            is_x (bool): set this to be the x-axis variable when plotting data, defaults to False
        """
        if not self.param_present(argv['name']):
            args = dict(argv)
            args['param_type'] = 'input'
            self.ecs.append(Measured_param(**args))
            is_x = argv.get('is_x', False)
            if is_x:
                self.set_ec_x(argv['name'])
        else:
            raise ValueError("It looks like you're trying to add a duplicate experimental condition, %s!"%argv['name'])

    def set_ec_x(self, param_name, verbose=False):
        """Set the x-variable for experimental conditions."""
        if not param_name in self.param_names('ec'):
            if verbose:
                print("Adding the variable %s to the list of experimental conditions and setting it as the x-axis variable for plotting."%param_name)
            self.add_ec(name=param_name)
        elif self.ec_x_name==param_name:
            pass #already done
        elif not self.ec_x_name==None:
            print("Overwriting previous ec_x variable (%s) with %s."%(self.ec_x_name,param_name))
        self.ec_x_name = param_name

    def get_ec_x(self):
        return self.find_param(self.ec_x_name)

    def add_output(self, **argv):
        """
        Add an output variable.

        Args:
            name (str): name of the parameter, required
            units (str): units in which parameter is measured (defaults to 'unitless')
            tolerance (float): smallest difference between two values of this parameter to consider "real," defaults to 1E-6
        """
        if not self.param_present(argv['name']):
            args = dict(argv)
            args['param_type'] = 'output'
            self.output.append(Measured_param(**args))
        else:
            raise ValueError("It looks like you're trying to add a duplicate output variable, %s!"%argv['name'])

    def param_names(self, param_type=None):
        """
        Return a list of parameter names. If no arguments provided, output will be a dict, if a type is provided, output will be a list of just the parameter names of that type.
        """
        if param_type=='fit':
            return [p.name for p in self.fit_params]
        elif param_type=='ec':
            return [c.name for c in self.ecs]
        elif param_type=='output':
            return [o.name for o in self.output]
        else:
            return {'fit_params': [p.name for p in self.fit_params],
                    'ec': [p.name for p in self.ec],
                    'output': [p.name for p in self.output]}

    def all_params(self):
        """Return a flat list of all parameters of any type."""
        return [p for p in self.fit_params+self.ecs+self.output]

    def param_present(self, name):
        """
        Check that the param name isn't already present in a list.

        Args:
            name (str): name to check for
        """
        all_names = [p.name for p in self.all_params()]
        if name in all_names:
            return True
        else:
            return False

    def find_param(self, name):
        """
        Return the Param object with the given name.

        Args:
            name (str): name to search for
        """
        return [p for p in self.all_params() if p.name==name][0]

    def vals_equal(self, param_name, val1, val2):
        """
        Compare two values of a given param.

        Args:
            param_name (str): name of parameter, must be in one of the lists
            val1, val2 (float): values to be compared

        Returns:
            True if abs(val1-val2) < tolerance of param_name
        """
        tol = self.find_param(param_name)['tolerance']
        if abs(val1-val2)<=tol:
            return True
        else:
            return False

    def set_tolerance(self, param_name, tol):
        """Set the tolerance value for the given parameter."""
        self.find_param(param_name).set_tolerance(tol)

    def is_empty(self):
        if len(self.all_params())==0:
            return True
        else:
            return False

    def as_dict(self):
        d = {'ec_x_name':self.ec_x_name}
        d['ecs'] = [c.__dict__ for c in self.ecs]
        d['fit_params'] = [p.__dict__ for p in self.fit_params]
        d['output'] = [o.__dict__ for o in self.output]
        return d

    def __str__(self):
        d = deepcopy(self.as_dict())
        for i in range(len(d['fit_params'])):
            d['fit_params'][i]['edges'] = str(d['fit_params'][i]['edges'])[:70]+'...'
            d['fit_params'][i]['vals'] = str(d['fit_params'][i]['vals'])[:70]+'...'
            d['fit_params'][i]['val_range'] = str(d['fit_params'][i]['val_range'])
        return json.dumps(d, indent=4)
