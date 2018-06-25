import numpy as np
import pandas as pd
import math
from scipy.stats import norm
from itertools import product
from copy import deepcopy
import matplotlib as mpl
import matplotlib.pyplot as plt

class Pmf(object):
    """
    Class that stores a PMF capable of nested sampling / "adaptive mesh refinement".

    Stores probabilities in a DataFrame which associates regions of parameter space with probability values.
    """

    def make_points_list(self, params, total_prob=1.0):
        """
        Helper function for Pmf.__init__ as well as Pmf.subdivide.
        Given names and values for parameters, generate DataFrame listing
        values, bounds, and probabilities.

        Args:
            params (:obj:`param_list`):
            total_prob (`float`): total probability to divide among points in parameter space - for an initialization, this is 1.0, for a subdivide call will be less.

        Returns:
            DataFrame with columns for each parameter's value, min, and max
            as well as a probability associated with that point in parameter
            space
        """

        # generate the list of box center coordinates
        param_indices = [range(len(p['vals'])) for p in params.fit_params]
        points = [p for p in product(*param_indices)]

        # iterate through to get values and add columns for param ranges
        for i in range(len(points)): # for every point (read: combination of value indices) in parameter space...
            row = []
            for j in range(len(params.fit_params)): # for each parameter...
                # get value
                row.append(params.fit_params[j]['vals'][points[i][j]])
                # get min and max of bounding box
                row.append([params.fit_params[j]['edges'][points[i][j]],params.fit_params[j]['edges'][points[i][j+1]]])
            points[i] = row

        points = np.array(points)
        param_names = [p['name'] for p in params.fit_params]
        columns = [c for c in np.append([p for p in param_names],[np.append([p+'_min'],[p+'_max']) for p in param_names])]

        df = pd.DataFrame(data=points, columns=columns)
        df['prob'] = [total_prob/len(df) for i in range(len(df))]

        return df


    def __init__(self, params):
        """
        Args:
            params (:obj:`param_list`): param_list object containing parameters to be fit and associated metadata
        """

        # for now just copy in the param_list wholesale
        # eventually should probably scrub and/or update vals...
        self.params = params.fit_params
        self.points = self.make_points_list(params)

    def normalize(self):
        """Normalize overall PMF."""
        norm_const = self.points['prob'].sum()
        #print(norm_const,type(norm_const))
        if self.equals_ish(float(norm_const),0):
            print("Somehow the normalization constant was zero! To play it safe, I won't do anything.")
        else:
            self.points['prob'] = [p/norm_const for p in self.points['prob']]

    def uniformize(self):
        """
        Keep PMF shape and subdivisions but make every probability equal.
        Useful for rerunning whole inference after subdividing.

        Note that because subdivisions are not uniform that this is NOT a uniform prior anymore.
        """
        norm_const = len(self.points)
        self.points['prob'] = [1.0/norm_const for i in range(norm_const)]

    def all_current_values(self, param):
        """
        List all values currently being considered for param.
        """
        ls = list(set(list(self.points[param])))
        ls.sort()
        return ls

    def find_neighbor_boxes(self, index):
        """
        Find and return all boxes neighboring the box at index.
        """
        all_vals = {param:self.all_current_values(param['name']) for param in self.params}
        this_point = self.points.iloc[index]
        indices_to_intersect=[]
        # get range of values of each param to consider "neighbors"
        for param in self.params:
            this_param_val = this_point[param]
            this_param_index = all_vals[param].index(this_param_val)

            # handle the edge cases
            if not this_param_index == 0:
                down_delta = all_vals[param][this_param_index]-all_vals[param][this_param_index-1]
            else:
                down_delta=0
            if not this_param_index == len(all_vals[param])-1:
                up_delta = all_vals[param][this_param_index+1]-all_vals[param][this_param_index]
            else:
                up_delta=0
            gt = self.points[param]>=this_param_val-down_delta
            lt = self.points[param]<=this_param_val+up_delta
            this_set = self.points[gt & lt]
            indices_to_intersect.append(set(this_set.index.values))

        indices = list(set.intersection(*indices_to_intersect))
        indices.sort()
        neighbor_points = self.points.iloc[indices]
        return neighbor_points

    
