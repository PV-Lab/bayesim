import numpy as np

class param_list(object):
    """
    Small class to facilitate listing parameters to be fit.

    Perhaps eventually this should also store fixed params and then it could be a standardized input format to the model function? Except you'd have to write an intermediate parser for e.g. a PC1D input file which would just be written back out again, so it should be flexible to be able to just hold a filepath or something too.
    """

    def __init__(self):
        self.fit_params=[]

    def add_fit_param(self,**argv):
        """
        Add a parameter to the list.

        Args:
            name (str): name of parameter (required)
            val_range (:obj:`list` of :obj:`float`): [min, max] (required)
            length (int): initial length of this parameter (defaults to 10)
            min_width(`float`): minimum box width for this parameter (defaults to 0.01 of total range)
            spacing (str): 'linear' or 'log' (defaults to linear)
            units (str): units for this parameter, if any (defaults to 'unitless')
        """
        # sanity checks
        assert 'name' in argv and 'val_range' in argv, "Parameter must at least have a name and a range!"
        assert len(val_range)==2 and val_range[1]>val_range[0], "val_range must be of length 2 with second entry larger than first!"

        # set some defaults
        spacing = argv.setdefault('spacing','linear')
        length = argv.setdefault('length',10)
        units = argv.setdefault('units','unitless')

        # read in info
        param_info = dict(argv)

        if 'min_width' not in argv.keys():
            if spacing == 'linear':
                min_width = 0.01*(val_range[1]-val_range[0])
            elif spacing == 'log':
                min_width = (val_range[1]/val_range[0])**0.01

        # compute edges and values
        if spacing == 'linear':
            edges_vals = np.linspace(val_range[0],val_range[1],2*length+1)
        elif spacing=='log':
            edges_vals = np.geomspace(val_range[0],val_range[1],2*length+1)
        edges = edges_vals[::2]
        vals = edges_vals[1:-1:2]
        param_info['edges'] = edges
        param_info['vals'] = vals

        self.fit_params.append(argv)

    def list_fit_param_vals()
        param_vals = {}
        for param in self.fit_params:
            if param['spacing']=='linear':
