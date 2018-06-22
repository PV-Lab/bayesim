class param_list(object):
    """
    Small class to facilitate listing parameters to be fit.
    """

    def __init__(self):
        self.params=[]

    def add_param(self,**argv):
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

        # set some defaults
        spacing = argv.setdefault('spacing','linear')
        length = argv.setdefault('length',10)
        units = argv.setdefault('units','unitless')
        if 'min_width' not in argv.keys():
            if argv['spacing'] == 'linear':
                min_width = 0.01*(val_range[1]-val_range[0])
            elif argv['spacing'] == 'log':
                min_width = (val_range[1]/val_range[0])**0.01

        self.params.append(argv)
