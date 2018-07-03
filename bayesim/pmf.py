import numpy as np
import pandas as pd
import math
from scipy.stats import norm
from itertools import product
from copy import deepcopy
import matplotlib as mpl
import matplotlib.pyplot as plt
import param_list as pl

class Pmf(object):
    """
    Class that stores a PMF capable of nested sampling / "adaptive mesh refinement".

    Stores probabilities in a DataFrame which associates regions of parameter space with probability values.

    Todo:
        make a save_state function for this
        (long-term) allow for non-gridded parameter space (Voronoi? Or MCMC with no subdivision)
    """

    def make_points_list(self, params, total_prob=1.0, new=False):
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
        param_indices = [range(len(p['vals'])) for p in params]
        points = [p for p in product(*param_indices)]
        # iterate through to get values and add columns for param ranges
        for i in range(len(points)): # for every point (read: combination of value indices) in parameter space...
            row = []
            for j in range(len(params)): # for each parameter...
                # get value
                val_index = points[i][j]
                row.append(params[j]['vals'][val_index])
                # get min and max of bounding box
                row.extend([params[j]['edges'][val_index],params[j]['edges'][val_index+1]])
            if new:
                row.extend([True])
            else:
                row.extend([False])
            points[i] = row

        points = np.array(points)
        param_names = [p['name'] for p in params]
        columns = [c for l in [[n,n+'_min',n+'_max'] for n in param_names] for c in l]+['new'] # there has to be a more readable way to do this

        df = pd.DataFrame(data=points, columns=columns)
        df['prob'] = [total_prob/len(df) for i in range(len(df))]

        return df


    def __init__(self, **argv):
        """
        Provide one argument or the other.

        Args:
            params (:obj:`param_list`): param_list object containing parameters to be fit and associated metadata
            param_points (`DataFrame`): DataFrame containing all parameter points to start with
        """

        if 'params' in argv.keys():
            # for now just copy in the param_list wholesale
            # eventually should probably scrub and/or update vals...
            self.params = argv['params'].fit_params # last dot might be wrong
            self.points = self.make_points_list(self.params)
        elif 'param_points' in argv.keys():
            # need to implement
            pass

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
        all_vals = {param['name']:self.all_current_values(param['name']) for param in self.params}
        this_point = self.points.iloc[index]
        indices_to_intersect=[]
        # get range of values of each param to consider "neighbors"
        for param in self.params:
            this_param_val = this_point[param['name']]
            this_param_index = all_vals[param['name']].index(this_param_val)

            # handle the edge cases
            if not this_param_index == 0:
                down_delta = all_vals[param['name']][this_param_index]-all_vals[param['name']][this_param_index-1]
            else:
                down_delta=0
            if not this_param_index == len(all_vals[param['name']])-1:
                up_delta = all_vals[param['name']][this_param_index+1]-all_vals[param['name']][this_param_index]
            else:
                up_delta=0
            gt = self.points[param['name']]>=this_param_val-down_delta
            lt = self.points[param['name']]<=this_param_val+up_delta
            this_set = self.points[gt & lt]
            indices_to_intersect.append(set(this_set.index.values))

        indices = list(set.intersection(*indices_to_intersect))
        indices.sort()
        neighbor_points = self.points.iloc[indices]
        return neighbor_points

    def subdivide(self, threshold_prob, include_neighbors=True):
        """
        Subdivide all boxes with P > threshold_prob and assign "locally uniform" probabilities within each box. If include_neighbors is true, also subdivide all boxes neighboring those boxes to facilitate flow of probability mass.

        For now, just divides into two along each direction. Ideas for improvement:
        * divide proportional to probability mass in that box such that minimum prob is roughly equal to maximum prob of undivided boxes
        * user-specified divisions along dimensions (including NOT dividing in a given direction)
        """
        num_divs = {p['name']:2 for p in self.params} #dummy for now

        # set all 'new' flags to False
        flags = [False] * len(self.points)
        self.points['new'] = flags

        # pick out the boxes that will be subdivided
        to_subdivide = self.points[self.points['prob']>threshold_prob]
        #print(len(to_subdivide))

        num_high_prob_boxes = len(to_subdivide)

        if num_high_prob_boxes == 0:
            print('nothing to subdivide!')
            return

        if include_neighbors:
            # find neighbor boxes
            neighbor_list = []
            for box in to_subdivide.iterrows():
                neighbor_list.append(self.find_neighbor_boxes(box[0]))
            neighbors = pd.concat(neighbor_list)
            to_subdivide = pd.concat([to_subdivide,neighbors])
            to_subdivide = to_subdivide.drop_duplicates(subset=self.points.columns[self.points.columns != 'prob']) # exclude probability when considering identical-ness

        num_nbs = len(to_subdivide)-num_high_prob_boxes

        # create new boxes (and delete old ones)
        new_boxes = []
        dropped_inds = []
        for box in to_subdivide.iterrows():
            # check if minimum width is already satisfied
            num_divs_here = num_divs
            for p in self.params:
                if p['spacing']=='linear':
                    this_width = box[1][p['name']+'_max']-box[1][p['name']+'_min']
                elif p['spacing']=='log':
                    this_width = box[1][p['name']+'_max']/box[1][p['name']+'_min']
                if this_width <= p['min_width']:
                    # don't divide along this direction
                    num_divs_here[p['name']] = 1
                    print('Minimum width/factor of ' + str(p['min_width']) + ' already satisfied for ' + p['name'] + ' at point: \n' + str(box[1]))

            # first, remove this box from DataFrame
            dropped_inds.append(box[0])
            self.points = self.points.drop([box[0]])

            # create new DataFrame with subdivided boxes
            new_pl = pl.param_list()
            for p in self.params:
                # copy same params except for ranges
                new_pl.add_fit_param(name=p['name'],val_range=[box[1][p['name']+'_min'],box[1][p['name']+'_max']],length=num_divs_here[p['name']],min_width=p['min_width'],spacing=p['spacing'],units=p['units'])
            # make new df, spreading total prob from original box among new smaller ones
            new_boxes.append(self.make_points_list(new_pl.fit_params,total_prob=box[1]['prob'],new=True))

        new_boxes = pd.concat(new_boxes)

        # concatenate new DataFrame to self.points
        self.points = pd.concat([self.points,new_boxes])

        # sort values
        self.points = self.points.sort_values([p['name'] for p in self.params])

        # reindex DataFrame
        self.points = self.points.reset_index(drop=True)

        # should be normalized already, but just in case:
        self.normalize()

        print(str(num_high_prob_boxes) + ' box(es) with probability > ' + str(threshold_prob) + ' and ' + str(num_nbs) + ' neighboring boxes subdivided!')

        return new_boxes, to_subdivide

    def multiply(self, other_pmf):
        """
        Compute and store renormalized product of this Pmf with other_pmf.
        """

        # check for silliness
        assert isinstance(other_pmf, Pmf), "You didn't feed in a Pmf object!"
        assert len(self.points) == len(other_pmf.points), "Pmf's are over different numbers of points. Can't exactly do a pointwise multiplication on that, can I?"
        # should add a check that all points match

        # copy and sort/index DataFrames to match so that we're multiplying
        # the right probabilities together
        these_probs = deepcopy(self.points)
        other_probs = deepcopy(other_pmf.points)

        cols_for_sort = [p['name'] for p in self.params]
        these_probs.sort_values(by=cols_for_sort)
        other_probs.sort_values(by=cols_for_sort)

        # actually multiply
        new_probs = these_probs['prob'] * other_probs['prob']

        # check if there's any probability there...
        if self.equals_ish(np.sum(new_probs),0):
            print("You're gonna have a bad time...not multiplying")
        else:
            self.points['prob'] = new_probs
            self.normalize()

    def likelihood(self, **argv):
        """
        Compute likelihood over this Pmf's parameter space assuming model_func() represents the true physical behavior of the system.

        Obviously, model_func() needs to be a callable in the working namespace and has to accept conditions in the format that they're fed into this function.

        Args:
            meas (`float`): one output value
            model_at_ec (`DataFrame`): DataFrame containing model data at the experimental condition of the measurement and error values in a column called 'error' for every point in parameter space
            output_col (`str`): name of column with output variable

            ec: dict with keys of condition names and values
            meas: one output value e.g. J
            error: error in measured value (stdev of a Gaussian)
            model_func: should accept one dict of params and one of conditions and output measurement

        Todo:
            alternative error models?
            allow feeding in a list of observations?
        """

        # read in and process inputs
        meas = argv['meas']
        model_data = argv['model_at_ec']
        model_data.sort_values([p['name'] for p in self.params])
        model_data.reset_index(drop=True)
        output_col = argv['output_col']

        # set up likelihood DF
        lkl = deepcopy(self)
        new_probs = np.zeros([len(lkl.points),1])

        # here's the actual loop that computes the likelihoods
        for point in lkl.points.iterrows():
            #print(ec, point[1])
            #model_val = model_func(ec, dict(point[1]))
            model_pt = model_data.iloc[point[0]]
            model_val = float(model_pt[output_col])
            err = float(model_pt['error'])
            new_probs[point[0]] = norm.pdf(meas, loc=model_val, scale=abs(err))

        # copy these values in
        lkl.points['prob'] = new_probs

        # make sure that the likelihood isn't zero everywhere...
        if self.equals_ish(np.sum(new_probs),0):
            print('likelihood has no probability! :( This happened at this measurement:\n', conditions)
        lkl.normalize()
        return lkl

    def most_probable(self, n):
        """Return the n largest probabilities in a new DataFrame.
        """

        sorted_probs = self.points.sort_values(by='prob',ascending=False)
        return sorted_probs.iloc[0:n]

    def equals_ish(self, num1, num2, thresh = 1e-12):
        """helper function to deal with machine error issues
        """

        if abs(num1-num2) < thresh:
            return True
        else:
            return False

    def inside_bounds(self, bin_lims, bounds):
        """
            Helper function for project_1D.

            Args:
                bin_lims (:obj:`list` of :obj:`float`): lower and upper bound of bin in question
                bounds (:obj:`list` of :obj:`float`): lower and upper bound of param point in question

            Returns:
                True if bin limits are equal to or inside point bounds, False otherwise
        """

        if (bin_lims[0] >= bounds[0] or self.equals_ish(bin_lims[0], bounds[0])) and (bin_lims[1] <= bounds[1] or self.equals_ish(bin_lims[1],bounds[1])):
            return True
        else:
            return False

    def length_fraction(self, bin_lims, bounds, log_spacing):
        """
        Another helper function for project_1D.

        Args:
            bin_lims (:obj:`list` of :obj:`float`): lower and upper bound of bin in question
            bounds (:obj:`list` of :obj:`float`): lower and upper bound of Param_point
            log_spacing (bool): whether spacing is logarithmic or not

        Returns:
            float: Fraction of prob that should be in this bin
        """

        if log_spacing:
            bin_length = bin_lims[1]/bin_lims[0]
            point_length = bounds[1]/bounds[0]
        else:
            bin_length = bin_lims[1]-bin_lims[0]
            point_length = bounds[1]-bounds[0]

        return bin_length/point_length

    def project_1D(self, param):

        """
        Project down to a one-dimensional PMF over the given parameter. Used by the visualize() method.

        Args:
            param (dict): one of `self.params`

        Returns:
            bins (:obj:`list` of :obj:`float`): bin edges for plotting with matplotlib.pyplot.hist (has length one more than next return list)
            probs (:obj:`list` of :obj:`float`): probability values for histogram-style plot - note that these technically have units of the inverse of whatever the parameter being plotted is (that is, they're probability densities)

        Todo:
            make it faster?
        """

        ## first find bin edges
        # pull all bounds, then flatten, remove duplicates, and sort
        bins = sorted(list(set(list(self.points[param['name']+'_min'])+list(self.points[param['name']+'_max']))))

        # set() doesn't seem to perfectly remove duplicates (#YayNumerics) so here's a hacky extra check
        for i in [len(bins)-j-1 for j in range(len(bins)-1)]: #count backwards
            if self.equals_ish(bins[i],bins[i-1]): del bins[i]

        ## now for every pair of bin edges, get the probability
        probs = np.zeros(len(bins)-1)
        if param['spacing']=='log':
            log_spaced=True
        elif param['spacing']=='linear':
            log_spaced=False

        for i in range(len(probs)):
            this_bin = [bins[i],bins[i+1]]
            for row in self.points.iterrows():
                this_point = (row[1][param['name']+'_min'],row[1][param['name']+'_max'])
                if self.inside_bounds(this_bin,this_point):
                    # take the appropriate fraction of the probability
                    prob_to_add = self.length_fraction(this_bin, this_point, log_spaced) * row[1]['prob']
                    probs[i] = probs[i] + prob_to_add

        ## finally, normalize according to length and scaling (log vs. linear)
        if log_spaced:
            bin_lengths = [bins[i+1]/bins[i] for i in range(len(bins)-1)]
            bins = [math.log(edge,10) for edge in bins]
        else:
            bin_lengths = [bins[i+1]-bins[i] for i in range(len(bins)-1)]

        probs = [probs[i]/bin_lengths[i] for i in range(len(probs))]
        # and normalize actual PMF
        norm = np.sum(probs)
        probs = [prob/norm for prob in probs]

        ## now go back to actual values
        if log_spaced:
            bins = [math.pow(10, b) for b in bins]

        return bins, probs

    def project_2D(self, x_param, y_param):

        """
        Project down to two dimensions over the two parameters. This one doesn't actually need to sum, it just draws a bunch of (potentially overlapping) rectangles with transparencies according to their probability densities (as a fraction of the normalized area). Used by the visualize() method.

        Args:
            x_param (dict): one of `self.params`, to be the x-axis of the 2D joint plot
            y_param (dict): one of `self.params`, to be the y-axis of the 2D joint plot

        Returns:
            (:obj:`list` of :obj:`matplotlib.patches.Rectangle`): patches for plotting the 2D joint probability distribution

        Todo:
            Maybe would speed up things if I passed the list of parameters for the patches rather than the objects themselves?
        """
        x_name = x_param['name']
        y_name = y_param['name']
        max_prob = max(self.points['prob'])
        patches = []
        for row in self.points.iterrows():
            x_min = row[1][x_name+'_min']
            x_width = row[1][x_name+'_max'] - x_min
            y_min = row[1][y_name+'_min']
            y_width = row[1][y_name+'_max'] - y_min
            alpha = row[1]['prob']/max_prob
            if alpha>1e-3: #this speeds it up a lot
                patches.append(mpl.patches.Rectangle((x_min,y_min),x_width,y_width,alpha=alpha))

        return patches

    def visualize(self, frac_points=1.0):
        """
        Make histogram matrix to visualize the PMF.

        Todo:
            include option to outline all boxes to see subdivisions
            figure out axis labeling and stuff for units, logs, etc.
            how to rescale - e.g. eliminating areas where prob less than some threshold?
            probably just add a bunch of optional arguments to handle these
            make it faster...
        """
        #start_time = timeit.default_timer()

        # find ranges to plot - this likely needs tweaking
        N = len(self.points)
        points_to_include = self.most_probable(int(frac_points*N))
        plot_ranges = {}
        for param in self.params:
            plot_ranges[param['name']] = [min(points_to_include[param['name']+'_min']), max(points_to_include[param['name']+'_max'])]

        fig, axes = plt.subplots(nrows=len(self.params), ncols=len(self.params), figsize=(10,9))

        #check1 = timeit.default_timer()
        #time1 = round(check1-start_time,2)
        #print('setup finished in ' + str(time1) + ' seconds')

        for rownum in range(0,len(self.params)):
            for colnum in range(0,len(self.params)):
                x_param = self.params[colnum]
                y_param = self.params[rownum]

                # pre-formatting
                axes[rownum][colnum].set_xlim(plot_ranges[x_param['name']][0],plot_ranges[x_param['name']][1])
                axes[rownum][colnum].set_axisbelow(True)

                for item in ([axes[rownum][colnum].xaxis.label, axes[rownum][colnum].yaxis.label] +axes[rownum][colnum].get_xticklabels() + axes[rownum][colnum].get_yticklabels()):
                    item.set_fontsize(20)

                if x_param['spacing']=='log':
                    axes[rownum][colnum].set_xscale('log')

                if rownum==colnum: #diagonal - single-variable histogram
                    #diag_start = timeit.default_timer()
                    bins, probs = self.project_1D(x_param)
                    #checkpoint = round(timeit.default_timer()-diag_start,2)
                    #print('project_1D took ' + str(checkpoint) + ' seconds')
                    if x_param['spacing']=='log':
                        vals = [math.sqrt(bins[i]*bins[i+1]) for i in range(len(probs))]
                    elif x_param['spacing']=='linear':
                        vals = [0.5*(bins[i]+bins[i+1]) for i in range(len(probs))]
                    axes[rownum][colnum].hist(vals, weights=probs, bins=bins, edgecolor='k', linewidth=1.0)
                    # formatting
                    axes[rownum][colnum].set_ylim(0,1)

                    #diag_finish = timeit.default_timer()
                    #diag_time = round(diag_finish-diag_start,2)
                    #print('diagonal plot finished in ' + str(diag_time) + ' seconds')

                elif rownum > colnum: # below diagonal
                    #offdiag_start = timeit.default_timer()
                    patches = self.project_2D(x_param, y_param)
                    #checkpoint = round(timeit.default_timer()-offdiag_start,2)
                    #print('project_2D took ' + str(checkpoint) + ' seconds')
                    for patch in patches:
                        axes[rownum][colnum].add_patch(patch)
                    # formatting
                    axes[rownum][colnum].set_ylim(plot_ranges[y_param['name']][0],plot_ranges[y_param['name']][1])
                    if y_param['spacing']=='log':
                        axes[rownum][colnum].set_yscale('log')
                    #offdiag_finish = timeit.default_timer()
                    #offdiag_time = round(offdiag_finish-offdiag_start,2)
                    #print('off-diagonal plot finished in ' + str(offdiag_time) + ' seconds')

                else: # above diagonal
                    fig.delaxes(axes[rownum][colnum])

        # put the labels on the outside
        for i in range(0,len(self.params)):
            xlabel = '%s [%s]' %(self.params[i]['name'],self.params[i]['units'])
            ylabel = '%s [%s]' %(self.params[i]['name'],self.params[i]['units'])
            axes[len(self.params)-1][i].set_xlabel(xlabel)
            if i>0: # top one is actually a probability
                axes[i][0].set_ylabel(ylabel)

        plt.tight_layout()
