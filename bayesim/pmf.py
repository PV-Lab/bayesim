import numpy as np
import pandas as pd
import math
from scipy.stats import norm,lognorm
from itertools import product
from copy import deepcopy
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import bayesim.params as pm
import timeit
from itertools import product

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
            params (:class:`.Param_list`):
            total_prob (float): total probability to divide among points in parameter space - for an initialization, this is 1.0, for a subdivide call will be less.

        Returns:
            :obj:DataFrame with columns for each parameter's value, min, and max
            as well as a probability associated with that point in parameter
            space
        """

        # generate the list of box center coordinates
        param_indices = [range(len(p.vals)) for p in params]
        points = [p for p in product(*param_indices)]
        # iterate through to get values and add columns for param ranges
        for i in range(len(points)): # for every point (read: combination of value indices) in parameter space...
            row = []
            for j in range(len(params)): # for each parameter...
                # get value
                val_index = points[i][j]
                row.append(params[j].vals[val_index])
                # get min and max of bounding box
                row.extend([params[j].edges[val_index],params[j].edges[val_index+1]])
            points[i] = row

        points = np.array(points)
        columns = [c for l in [[n,n+'_min',n+'_max'] for n in self.param_names()] for c in l] # there has to be a more readable way to do this

        df = pd.DataFrame(data=points, columns=columns)
        df['prob'] = [total_prob/len(df) for i in range(len(df))]

        return df


    def __init__(self, **argv):
        """
        Initialize a Pmf object. Provide one of the three input options, or else will initialize empty.

        Args:
            params (:obj:`Param_list`): Param_list object containing parameters to be fit and associated metadata
            param_points (:obj:`DataFrame`): DataFrame containing all parameter points to start with
            prob_dict (:obj:`dict`): output of as_dict()
        """

        if 'prob_dict' in argv.keys():
            prob_dict = argv['prob_dict']
            self.is_empty = prob_dict['is_empty']
            self.num_sub = prob_dict['num_sub']
            self.points = prob_dict['points']
            self.params = [pm.Fit_param(**p) for p in prob_dict['params']]
        elif 'params' in argv.keys():
            # for now just copy in the param_list wholesale
            # eventually should probably scrub and/or update vals...
            self.num_sub = 0
            self.params = argv['params']
            self.points = self.make_points_list(self.params)
            if len(self.params)==0:
                self.is_empty = True
            else:
                self.is_empty = False
        elif 'param_points' in argv.keys():
            raise NotImplementedError('Still need to implement initializing a Pmf from a list of points!')
        else: # empty arguments
            self.num_sub = 0
            self.params = []
            self.points = pd.DataFrame()
            self.is_empty = True

    def as_dict(self):
        """Return this Pmf object in (readable) dictionary form."""
        d = deepcopy(self.__dict__)
        d['params'] = [p.__dict__ for p in self.params]
        return d

    def normalize(self, **argv):
        """Normalize overall PMF."""
        norm_const = self.points['prob'].sum()
        verbose = argv.get('verbose', False)
        #print(norm_const,type(norm_const))
        if abs(norm_const)<1e-12:
            if verbose:
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
        List all values currently being considered for `param`.
        """
        ls = list(set(list(self.points[param])))
        ls.sort()
        return ls

    def find_neighbor_boxes(self, index):
        """
        Find and return all boxes neighboring the box at `index`.
        """
        all_vals = {name:self.all_current_values(name) for name in self.param_names()}
        param_spacing = {param.name:param.spacing for param in self.params}
        this_point = self.points.iloc[index]
        indices_to_intersect=[]
        # get range of values of each param to consider "neighbors"
        for p in self.params:
            param = p.name
            this_param_val = this_point[param]
            this_param_index = all_vals[param].index(this_param_val)

            # handle the edge cases
            if not this_param_index == 0:
                #down_delta = (all_vals[param['name']][this_param_index]-all_vals[param['name']][this_param_index-1])*1.001
                if param_spacing[param]=='linear':
                    down_delta = (this_point[param+'_max']-this_point[param+'_min'])*1.501
                elif param_spacing[param]=='log':
                    down_delta = this_point[param]-this_point[param+'_min']*(this_point[param+'_min']/(this_point[param+'_max']*1.001))
            else:
                down_delta=0
            if not this_param_index == len(all_vals[param])-1:
                #up_delta = (all_vals[param['name']][this_param_index+1]-all_vals[param['name']][this_param_index])*1.001
                if param_spacing[param]=='linear':
                    up_delta = (this_point[param+'_max']-this_point[param+'_min'])*1.501
                elif param_spacing[param]=='log':
                    up_delta = this_point[param+'_max']*(this_point[param+'_max']*1.001/this_point[param+'_min'])-this_point[param]
            else:
                up_delta=0
            #print(param,up_delta,down_delta)
            gt = self.points[param]>=this_param_val-down_delta
            lt = self.points[param]<=this_param_val+up_delta
            this_set = self.points[gt & lt]
            #print(this_set[[pm['name'] for pm in self.params]])
            indices_to_intersect.append(set(this_set.index.values))

        indices = list(set.intersection(*indices_to_intersect))
        indices.sort()
        neighbor_points = self.points.iloc[indices]
        # check if we went too far out along any axis
        inds_to_drop = []
        all_out_query = ''
        for p in self.params:
            param = p.name
            other_params = [pm.name for pm in self.params if not pm.name==param]
            # clunky brute-force search but it should be a smallish list so hopefully it won't kill us
            # first check that we're inside the box for all other params
            in_box_query = ''
            for q in other_params:
                if param_spacing[q]=='linear':
                    in_box_query = in_box_query + '%s<%f & %s>%f & '%(q,this_point[q+'_max'],q,this_point[q+'_min'])
                elif param_spacing[q]=='log':
                    in_box_query = in_box_query + '%s<%E & %s>%E & '%(q,this_point[q+'_max'],q,this_point[q+'_min'])
            # then check each direction in this param
            if param_spacing[param]=='linear':
                gt_query = in_box_query + '%s>%f'%(param,this_point[param+'_max'])
                lt_query = in_box_query + '%s<%f'%(param,this_point[param+'_min'])
                all_out_query = all_out_query + '(%s>%f | %s<%f) & '%(param,this_point[param+'_max'],param,this_point[param+'_min'])
            elif param_spacing[param]=='log':
                gt_query = in_box_query + '%s>%E'%(param,this_point[param+'_max'])
                lt_query = in_box_query + '%s<%E'%(param,this_point[param+'_min'])
                all_out_query = all_out_query + '(%s>%E | %s<%E) & '%(param,this_point[param+'_max'],param,this_point[param+'_min'])
            # pull points that are outside bounds of this param
            gt_check = neighbor_points.query(gt_query)
            lt_check = neighbor_points.query(lt_query)
            # group them by coords in other params
            gt_check_grps = gt_check.groupby(by=other_params)
            lt_check_grps = lt_check.groupby(by=other_params)
            # if any group has multiple indices, drop the ones corresponding to the "further out" values
            for grp,inds in gt_check_grps:
                if len(inds)>1:
                    vals = gt_check.loc[inds.index][param]
                    keep = vals.idxmin()
                    drop = [i for i in inds.index if not i==keep]
                    inds_to_drop.extend(drop)
                    #print('dropping '+str(drop))
            for grp,inds in lt_check_grps:
                if len(inds)>1:
                    vals = lt_check.loc[inds.index][param]
                    keep = vals.idxmax()
                    drop = [i for i in inds.index if not i==keep]
                    inds_to_drop.extend(drop)
                    #print('dropping '+str(drop))
        # and finally, check for ones that aren't inside bounds at all
        # for now, drop all of them - should keep closest corner really
        all_out_query = all_out_query[:-3]
        all_out_check = neighbor_points.query(all_out_query)
        inds_to_drop.extend(all_out_check.index)
        return neighbor_points.drop(labels=inds_to_drop)

    def subdivide(self, threshold_prob, include_neighbors=True, **argv):
        """
        Subdivide all boxes with P > threshold_prob and assign "locally uniform" probabilities within each box. If include_neighbors is true, also subdivide all boxes neighboring those boxes.

        Boxes with P < threshold_prob are deleted.

        Args:
            threshold_prob (`float`): probability above which a box should be retained
            include_neighbors (`bool`): whether to also subdivide all immediate neighbors to boxes meeting the threshold
            verbose (`bool`): Verbosity, default false
        """
        num_divs = {p.name:2 for p in self.params} #dummy for now

        verbose = argv.get('verbose', False)
        
        # pick out the boxes that will be subdivided
        to_subdivide = self.points[self.points['prob']>threshold_prob]
        #print(len(to_subdivide))
        #dropped_boxes = deepcopy(to_subdivide)

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

        # check if minimum width is already satisfied for any parameter
        test_box = to_subdivide.iloc[0]
        for p in self.params:
            if p.spacing=='linear':
                this_width = box[1][p.name+'_max']-box[1][p.name+'_min']
            elif p.spacing=='log':
                this_width = box[1][p.name+'_max']/box[1][p.name+'_min']
            if this_width <= p.min_width:
                # don't divide along this direction
                num_divs[p.name] = 1
                if verbose:
                    print('Minimum width/factor of %s already satisfied for parameter %s, not subdividing in that direction!'%(p.get_val_str(p.min_width), p.name))

        # create new boxes (and delete old ones)
        new_boxes = []
        #dropped_inds = []
        for box in to_subdivide.iterrows():
            # first, remove this box from DataFrame
            #dropped_inds.append(box[0])
            self.points.drop([box[0]], inplace=True)

            # create new DataFrame with subdivided boxes
            new_pl = pm.Param_list()
            for p in self.params:
                # copy same params except for ranges
                new_pl.add_fit_param(name=p.name,
                val_range=[box[1][p.name+'_min'], box[1][p.name+'_max']], length=num_divs[p.name], min_width=p.min_width, spacing=p.spacing, units=p.units, tolerance=p.tolerance)
            # make new df, spreading total prob from original box among new smaller ones
            new_boxes.append(self.make_points_list(new_pl.fit_params, total_prob=box[1]['prob']))

        # put in the new points (and completely drop the old ones)
        self.points = pd.concat(new_boxes)

        # make new lists of self.params (this way might be slow...)
        new_params = pm.Param_list()
        for p in self.params:
            p_args = p.__dict__
            p_args['vals'] = vals=list(set(list(self.points[p.name])))
            del p_args['val_range']
            #print(p_args)
            #new_params.add_fit_param(name=p.name, vals=list(set(list(self.points[p.name]))), spacing=p.spacing)
            new_params.add_fit_param(**p_args)
        self.params = [p for p in new_params.fit_params]

        # sort values
        self.points = self.points.sort_values(self.param_names())

        # reindex DataFrame
        self.points = self.points.reset_index(drop=True)

        # should be normalized already, but just in case:
        self.normalize(verbose=verbose)

        # increment subdivide count
        self.num_sub = self.num_sub + 1

        print(str(num_high_prob_boxes) + ' box(es) with probability > ' + str(threshold_prob) + ' and ' + str(num_nbs) + ' neighboring boxes subdivided!')

        return new_boxes

    def multiply(self, other_pmf, **argv):
        """
        Compute and store renormalized product of this Pmf with other_pmf.

        Args:
            other_pmf (:class:`.Pmf`): PMF to multiply by
        """

        verbose = argv.get('verbose', False)
        
        # check for silliness
        assert isinstance(other_pmf, Pmf), "You didn't feed in a Pmf object!"
        assert len(self.points) == len(other_pmf.points), "Pmf's are over different numbers of points. Can't exactly do a pointwise multiplication on that, can I?"
        # should add a check that all points match

        # copy and sort/index DataFrames to match so that we're multiplying
        # the right probabilities together
        these_probs = deepcopy(self.points)
        other_probs = deepcopy(other_pmf.points)

        these_probs.sort_values(by=self.param_names())
        other_probs.sort_values(by=self.param_names())

        # actually multiply
        new_probs = these_probs['prob'] * other_probs['prob']

        # check if there's any probability there...
        if abs(np.sum(new_probs))<1e-12:
            if verbose:
                print("You're gonna have a bad time...not multiplying")
        else:
            self.points['prob'] = new_probs
            self.normalize(verbose=verbose)

    def likelihood(self, **argv):
        """
        Compute likelihood over this Pmf's parameter space given modeled data at the given EC's for every parameter space point and a measurement at the same EC's.

        Args:
            meas (`float`): one output value
            model_at_ec (:py:class:`DataFrame`): DataFrame containing model data at the experimental condition of the measurement and uncertainty values in a column called 'uncertainty' for every point in parameter space
            output_col (`str`): name of column with output variable

            ec: dict with keys of condition names and values
            meas: one output value e.g. J
            unc: uncertainty in measured value (stdev of a Gaussian)
            model_func: should accept one dict of params and one of conditions and output measurement (might deprecate)
            verbose (`bool`): be verbose or not
        """

        # read in and process inputs
        meas = argv['meas']
        model_data = argv['model_at_ec']
        model_data.sort_values(self.param_names())
        model_data.reset_index(drop=True)
        output_col = argv['output_col']
        meas_val = meas[output_col]
        meas_err = meas['uncertainty']
        verbose = argv.get('verbose', False)

        # set up likelihood DF
        lkl = deepcopy(self)
        new_probs = np.zeros([len(lkl.points),1])

        delta_count = 0
        nan_count = 0
        skip_count = 0
        
        # check that lengths match
        if len(lkl.points)==len(model_data):
            # here's the actual loop that computes the likelihoods
            for point in lkl.points.iterrows():
                #print(ec, point[1])
                #model_val = model_func(ec, dict(point[1]))
                model_pt = model_data.iloc[point[0]]
                model_val = float(model_pt[output_col])
                if not np.isnan(model_val):
                    model_err = float(model_pt['uncertainty'])
                    if np.isnan(model_err):
                        err = 2*meas_err #janky but whatev
                    else:
                        err = model_err + meas_err

                    # tally how many times deltas were bigger
                    if model_err > meas_err:
                        delta_count = delta_count + 1

                    new_probs[point[0]] = norm.pdf(meas_val, loc=model_val, scale=abs(err))
                else:
                    new_probs[point[0]] = 1.0/len(lkl.points)
                    nan_count = nan_count + 1
            
            # copy these values in
            lkl.points['prob'] = new_probs

            # make sure that the likelihood isn't zero everywhere...
            if abs(np.sum(new_probs))<1e-12:
                if verbose:
                    print('likelihood has no probability! :(')
                skip_count = skip_count + 1
            if any(np.isnan(np.array(self.points['prob']))):
                print(self.points[self.points.isnull().any(axis=1)])
                raise ValueError('Uh-oh, some probability is NaN!')

        else:
            if verbose:
                print('Missing model data for ')
                print(meas)
                print('Skipping likelihood calculation here.\n')

        lkl.normalize()        
        return lkl, delta_count, nan_count, skip_count

    def most_probable(self, n):
        """Return the n largest probabilities in a new DataFrame.
        """
        sorted_probs = self.points.sort_values(by='prob',ascending=False)
        return sorted_probs.iloc[0:n]

    def param_names(self):
        """Return list of parameter names of this PMF."""
        return [p.name for p in self.params]

    def populate_dense_grid(self,**argv):
        """
        Populate a grid such as the one created by make_dense_grid.

        Args:
            df (`obj`:DataFrame): DataFrame to populate from (should have columns for every param)
            col_to_pull (`str`): name of the column to use when populating grid points
            make_ind_lists (`bool`): whether to return a list of indices corresponding to the first in every slice (used by bayesim.model.calc_model_gradients)

        Returns:
            a dict with keys for each thing requested
        """
        # read in inputs
        df = argv['df']
        col_to_pull = argv['col_to_pull']
        make_ind_lists = argv.get('make_ind_lists',True)

        # initialize things
        mat_shape = []
        pvals_indices = {}
        param_edges = {}

        for param in self.params:
            mat_shape.append(param.length)
            pvals_indices[param.name] = {param.vals[i]:i for i in range(len(param.vals))}
            #indices_lists.append(range(len(param_vals[pname])))

        mat = np.full(mat_shape,np.nan)

        # initialize optional things
        if make_ind_lists:
            ind_lists = {p.name:[] for p in self.params}

        for pt in df.iterrows():
            #print(pt)
            slices = []
            param_point = self.points.loc[pt[0]] # if df isn't self.points
            for p in self.params:
                min_val = param_point[p.name+'_min']
                max_val = param_point[p.name+'_max']
                inds = [pvals_indices[p.name][v] for v in p.vals if v>min_val and v<max_val]
                #print(p.name, min_val, max_val, p.vals, inds)
                slices.append(slice(min(inds),max(inds)+1,None))
                if make_ind_lists:
                    ind_lists[p.name].append(inds[0])
            if col_to_pull == 'prob':
                val = param_point['prob']
            else:
                val = pt[1][col_to_pull]
            mat[tuple(slices)] = val

        return_dict = {'mat':mat}
        if make_ind_lists:
            return_dict['ind_lists'] = ind_lists

        return return_dict

    def weighted_avgs(self):
        """Returns a dict with the weighted average for each parameter."""
        avgs = {}
        for param in self.params:
            bins, probs = self.project_1D(param)
            vals = param.vals
            avgs[param.name] = np.sum([probs[i]*vals[i] for i in range(len(vals))])
        return avgs


    def project_1D(self, param, dense_grid=[]):

        """
        Project down to a one-dimensional PMF over the given parameter. Used by the visualize() method.

        Args:
            param (:obj:`Fit_param`): one of `self.params`

        Returns:
            bins (:obj:`list` of :obj:`float`): bin edges for plotting with matplotlib.pyplot.hist (has length one more than next return list)
            probs (:obj:`list` of :obj:`float`): probability values for histogram-style plot - note that these technically have units of the inverse of whatever the parameter being plotted is (that is, they're probability densities)
            dense_grid (matrix): optionally, pass precomputed dense grid to save time
        """
        ## first find bin edges
        # pull all bounds, then flatten, remove duplicates, and sort
        bins = sorted(list(set(list(self.points[param.name+'_min'])+list(self.points[param.name+'_max']))))

        if len(dense_grid)==0:
            # generate dense grid and populate with probabilities
            dense_grid = self.populate_dense_grid(df=self.points, col_to_pull='prob', make_ind_lists=False)
            mat = dense_grid['mat']
        else:
            mat = dense_grid

        # sum along all dimensions except the parameter of interest
        param_ind = self.param_names().index(param.name)
        inds_to_sum_along = tuple([i for i in range(len(mat.shape)) if not i==param_ind])
        probs = np.nansum(mat,axis=inds_to_sum_along)

        return bins, probs

    def project_2D(self, x_param, y_param, no_probs=False, dense_grid=[]):

        """
        Project down to two dimensions over the two parameters. This one doesn't actually need to sum, it just draws a bunch of (potentially overlapping) rectangles with transparencies according to their probability densities (as a fraction of the normalized area). Used by the visualize() method.

        Args:
            x_param (dict): one of `self.params`, to be the x-axis of the 2D joint plot
            y_param (dict): one of `self.params`, to be the y-axis of the 2D joint plot
            no_probs (bool): whether to just show the grid boxes or include probability as determiner of transparency
            dense_grid (matrix): optionally, pass precomputed dense grid to save time

        Returns:
            (:obj:`list` of :obj:`matplotlib.patches.Rectangle`): patches for plotting the 2D joint probability distribution
        """
        if len(dense_grid)==0:
            # generate dense grid and populate with probabilities
            dense_grid = self.populate_dense_grid(df=self.points, col_to_pull='prob', make_ind_lists=False)
            mat = dense_grid['mat']
        else:
            mat = dense_grid

        max_prob = max(self.points['prob'])
        #param_vals = dense_grid['param_vals']
        param_edges = {p.name:p.edges for p in self.params}

        # sum along all dimensions except the parameter of interest
        inds_to_sum_along = tuple([i for i in range(len(mat.shape)) if not self.params[i].name in [x_param.name,y_param.name]])
        dense_probs = np.nansum(mat, axis=inds_to_sum_along)

        if no_probs:
            # generate list of patch parameters - first need every pair of indices
            ind_pairs = product(*[range(i) for i in dense_probs.shape])
            patches = []
            for pr in ind_pairs:
                x_min = param_edges[x_param.name][pr[0]]
                x_width = param_edges[x_param.name][pr[0]+1] - x_min
                y_min = param_edges[y_param.name][pr[1]]
                y_width = param_edges[y_param.name][pr[1]+1] - y_min
                if not dense_probs[pr]==np.nan:
                    patches.append(mpl.patches.Rectangle((x_min,y_min), x_width, y_width, fill=False, ec='k'))

            return patches

        else:
            return dense_probs

    def visualize(self, **argv):
        """
        Make histogram matrix to visualize the PMF.

        Args:
            frac_points (`float`): number >0 and <=1 indicating fraction of total points to visualize (will take the most probable, defaults to 1.0)
            just_grid (`bool`): whether to show only the grid (i.e. visualize subdivisions) or the whole PMF (defaults to False)
            fpath (`str`): optional, path to save image to
            true_vals (`dict`): optional, set of param values to highlight on PMF
            return_plots (bool): whether to return figure and axes (used by visualize_PMF_sequence in utils), default False
            color_index (int): from 0 to 4, choose from blue, green, red, purple, orange (defaults to 0)
        """
        # read in options
        frac_points = argv.get('frac_points', 1.0)
        just_grid = argv.get('just_grid', False)
        return_plots = argv.get('return_plots', False)
        if return_plots:
            plt.ioff()
            data = {i:{} for i in range(len(self.params))}
        color_index = argv.get('color_index',0)

        if 'fpath' in argv.keys():
            fpath = argv['fpath']

        if 'true_vals' in argv.keys():
            # check that all params are there
            true_vals = argv['true_vals']
            if not set(true_vals.keys())==set(self.param_names()):
                print('Your true_vals do not have all the paramter names! Proceeding without them.')
                plot_true_vals = False
            # check that values are within ranges
            elif not all(true_vals[p.name]>p.edges[0] and true_vals[p.name]<p.edges[-1] for p in self.params):
                print('Your true_vals are not within the correct bounds. Proceeding without them.')
                plot_true_vals = False
            else:
                plot_true_vals = True
        else:
            plot_true_vals = False

        # get color cycle and colormaps (just pull five)
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color'][:5]
        cmap_name_list = ['Blues','Greens','Reds','Purples','Oranges']
        color = colors[color_index]
        cmap_name = cmap_name_list[color_index]

        start_time = timeit.default_timer()

        # find ranges to plot - this likely needs tweaking
        N = len(self.points)
        points_to_include = self.most_probable(int(frac_points*N))
        plot_ranges = {}
        for param in self.params:
            plot_ranges[param.name] = [min(points_to_include[param.name+'_min']), max(points_to_include[param.name+'_max'])]

        fig, axes = plt.subplots(nrows=len(self.params), ncols=len(self.params), figsize=(5*len(self.params),5*len(self.params)))

        check1 = timeit.default_timer()
        time1 = round(check1-start_time,2)
        #print('setup finished in ' + str(time1) + ' seconds')

        dense_probs = self.populate_dense_grid(df=self.points, col_to_pull='prob', make_ind_lists=False, return_edges=True)['mat']

        param_ticks = self.pick_ticks(plot_ranges)

        for rownum in range(0,len(self.params)):
            for colnum in range(0,len(self.params)):
                x_param = self.params[colnum]
                y_param = self.params[rownum]

                # pre-formatting
                x_min = plot_ranges[x_param.name][0]
                x_max = plot_ranges[x_param.name][1]
                axes[rownum][colnum].set_xlim(x_min, x_max)
                axes[rownum][colnum].set_axisbelow(True)

                for item in ([axes[rownum][colnum].xaxis.label, axes[rownum][colnum].yaxis.label] +axes[rownum][colnum].get_xticklabels() + axes[rownum][colnum].get_yticklabels()):
                    item.set_fontsize(20)

                if rownum==colnum: #diagonal - single-variable histogram
                    if just_grid:
                        fig.delaxes(axes[rownum][colnum])
                    else:
                        diag_start = timeit.default_timer()
                        bins, probs = self.project_1D(x_param, dense_grid=deepcopy(dense_probs))
                        checkpoint = round(timeit.default_timer()-diag_start,2)
                        #print('project_1D took ' + str(checkpoint) + ' seconds')
                        if x_param.spacing=='log':
                            axes[rownum][colnum].set_xscale('log')
                            xtick_locs = [10**tick for tick in param_ticks[x_param.name]['locs']]
                            vals = [math.sqrt(bins[i]*bins[i+1]) for i in range(len(probs))]
                        elif x_param.spacing=='linear':
                            xtick_locs = param_ticks[x_param.name]['locs']
                            vals = [0.5*(bins[i]+bins[i+1]) for i in range(len(probs))]

                        # plot the data
                        axes[rownum][colnum].hist(vals, weights=probs, bins=bins, edgecolor='k', linewidth=1.0, color=color)
                        if return_plots:
                            data[rownum][colnum]={'vals':vals,'weights':probs,'bins':bins,'color':color}

                        # formatting...
                        # turn the ticks off and on (was having some stray minor ticks show up sometimes)
                        axes[rownum][colnum].tick_params(axis='x',which='both',bottom=False,labelbottom=False)
                        axes[rownum][colnum].tick_params(axis='x',which='major',bottom=True,labelbottom=True)
                        axes[rownum][colnum].set_xticks(xtick_locs, minor=False)
                        axes[rownum][colnum].set_xticklabels(param_ticks[x_param.name]['labels'])
                        axes[rownum][colnum].set_ylim(0,1)
                        axes[rownum][colnum].yaxis.set_label_position("right")
                        axes[rownum][colnum].yaxis.tick_right()
                        for label in axes[rownum][colnum].get_yticklabels():
                            label.set_fontsize(20)
                        axes[rownum][colnum].set_ylabel('P(%s)'%x_param.display_name, rotation=270, labelpad=20) #labelpad is kind of a brute-force way to do this and might break if we change the figure size, but va='bottom' wasn't working
                        #axes[rownum][colnum].`grid`(axis='y')

                        # add true value if desired
                        if plot_true_vals:
                            true_x = [true_vals[x_param.name]]
                            axes[rownum][colnum].scatter(true_x,0.05,200,'#FFFF00',marker='*',zorder=100)

                        diag_finish = timeit.default_timer()
                        diag_time = round(diag_finish-diag_start,2)
                        #print('diagonal plot finished in ' + str(diag_time) + ' seconds')

                elif rownum > colnum: # below diagonal
                    offdiag_start = timeit.default_timer()
                    # formatting axes ranges and ticks
                    if x_param.spacing=='log':
                        axes[rownum][colnum].set_xscale('linear')
                        x_min = np.log10(plot_ranges[x_param.name][0])
                        x_max = np.log10(plot_ranges[x_param.name][1])
                        axes[rownum][colnum].set_xlim([x_min, x_max])
                    if y_param.spacing=='log':
                        y_min = np.log10(plot_ranges[y_param.name][0])
                        y_max = np.log10(plot_ranges[y_param.name][1])
                    else:
                        y_min = plot_ranges[y_param.name][0]
                        y_max = plot_ranges[y_param.name][1]
                    axes[rownum][colnum].set_xticks(param_ticks[x_param.name]['locs'])
                    axes[rownum][colnum].set_xticklabels(param_ticks[x_param.name]['labels'])
                    axes[rownum][colnum].set_yticks(param_ticks[y_param.name]['locs'])
                    axes[rownum][colnum].set_yticklabels(param_ticks[y_param.name]['labels'])
                    axes[rownum][colnum].set_ylim([y_min, y_max])


                    if just_grid:
                        patches = self.project_2D(x_param, y_param, no_probs=True)
                        axes[rownum][colnum].grid(False)
                        for patch in patches:
                            axes[rownum][colnum].add_patch(patch)
                    else:
                        dense_probs_here = self.project_2D(x_param, y_param, dense_grid=deepcopy(dense_probs))


                        # generating the colormap and transparency
                        cmap = plt.get_cmap(cmap_name)
                        probs = dense_probs_here.transpose()
                        scaled_probs = probs/np.amax(probs)
                        alpha_weight = 0.5+0.1*color_index
                        alphas = alpha_weight*scaled_probs + (1-alpha_weight)*probs
                        colors = mpl.colors.Normalize(0.0, 1.0)(alphas)
                        colors = cmap(colors)
                        colors[..., -1] = alphas
                        # plot the data
                        axes[rownum][colnum].imshow(colors, aspect='auto', origin='lower', extent=(x_min, x_max, y_min, y_max), zorder=1)
                        if return_plots:
                            data[rownum][colnum]={'image':colors,'extent':(x_min, x_max, y_min, y_max)}

                    checkpoint = round(timeit.default_timer()-offdiag_start,2)
                    #print('project_2D took ' + str(checkpoint) + ' seconds')

                    if plot_true_vals:
                        true_x = [true_vals[x_param.name]]
                        true_y = [true_vals[y_param.name]]
                        if x_param.spacing=='log':
                            true_x = np.log10(true_x)
                        if y_param.spacing=='log':
                            true_y = np.log10(true_y)
                        axes[rownum][colnum].scatter(true_x,true_y,200,c="None",marker='o',linewidths=3,edgecolors='#FFFF00',zorder=100)
                    #axes[rownum][colnum].grid(False)
                    offdiag_finish = timeit.default_timer()
                    offdiag_time = round(offdiag_finish-offdiag_start,2)
                    #print('off-diagonal plot finished in ' + str(offdiag_time) + ' seconds')

                else: # above diagonal
                    fig.delaxes(axes[rownum][colnum])

        # put the labels on the outside
        for i in range(0,len(self.params)):
            xlabel = '%s [%s]' %(self.params[i].display_name, self.params[i].units)
            ylabel = '%s [%s]' %(self.params[i].display_name, self.params[i].units)
            axes[len(self.params)-1][i].set_xlabel(xlabel)
            if i>0: # top one is actually a probability
                axes[i][0].set_ylabel(ylabel)

        plt.tight_layout()

        if 'fpath' in argv.keys():
            plt.savefig(fpath)

        if return_plots:
            return {'fig':fig, 'axes':axes, 'data':data}
            plt.close()

        else:
            plt.show()


    def pick_ticks(self, plot_ranges):
        """Helper function for visualize."""
        param_ticks = {}
        for p in self.params:
            param_ticks[p.name] = {'locs':[], 'labels':[]}
            min_val = plot_ranges[p.name][0]
            max_val = plot_ranges[p.name][1]
            if p.spacing=='linear':
                p_range = max_val-min_val
                tick_spacing = round(p_range/4.5, -1*int(round(np.log10(0.1*p_range))))
                first_tick = int(min_val/tick_spacing)*tick_spacing
                if first_tick < min_val:
                    first_tick = first_tick+tick_spacing
                tick_locs = np.arange(first_tick, 0.01*p_range+max_val, tick_spacing)
                param_ticks[p.name]['locs'] = tick_locs
                if tick_spacing>2:
                    param_ticks[p.name]['labels'] = [str(int(loc)) for loc in tick_locs]
                else:
                    num_digits = int(round(-np.log10(tick_locs[-1]-tick_locs[0])+1))
                    param_ticks[p.name]['labels'] = [str(round(loc,num_digits)) for loc in tick_locs]
            elif p.spacing=='log':
                p_range = np.log10(max_val)-np.log10(min_val)
                if p_range<1.0:
                    tick_spacing=0.2
                elif p_range<3.0:
                    tick_spacing = 0.5
                elif p_range<=6.0:
                    tick_spacing = 1.0
                else:
                    tick_spacing = 2.0
                first_tick = int(np.log10(min_val)/tick_spacing)*tick_spacing
                if first_tick < np.log10(min_val):
                    first_tick = first_tick + tick_spacing
                tick_locs = np.arange(first_tick, 0.01*p_range+np.log10(max_val), tick_spacing)
                param_ticks[p.name]['locs'] = tick_locs
                if tick_spacing<1.0:
                    param_ticks[p.name]['labels'] = [r'10$^{%s}$'%str(round(loc,1)) for loc in tick_locs]
                else:
                    param_ticks[p.name]['labels'] = [r'10$^{%s}$'%int(loc) for loc in tick_locs]
        return param_ticks
