from copy import deepcopy
import numpy as np
import pandas as pd

def calc_deltas(grp, inds, param_lengths, model_data, fit_param_names, probs, output_var):
    # construct matrix of output_var({fit_params})
    subset = deepcopy(model_data.loc[inds])
    # sort and reset index of subset to match probs so we can use the find_neighbor_boxes function if needed
    subset.drop_duplicates(subset=fit_param_names, inplace=True)
    subset.sort_values(fit_param_names, inplace=True)
    subset.reset_index(inplace=True)
    if not len(subset.index)==len(probs.points.index):
        raise ValueError("Subset at EC's %s does not match probability grid!"%str(grp))

    # check if on a grid
    if not len(subset)==np.product(param_lengths):
        is_grid = False
        # construct grid at the highest level of subdivision
        dense_grid = probs.populate_dense_grid(df=subset, col_to_pull=output_var, make_ind_lists=True)
        mat = dense_grid['mat']
        ind_lists = dense_grid['ind_lists']

    else:
        is_grid = True
        mat = np.reshape(list(subset[output_var]), param_lengths)

    # given matrix, compute largest differences along any direction
    winner_dim = [len(mat.shape)]
    winner_dim.extend(mat.shape)
    winners = np.zeros(winner_dim)

    # for every dimension (fitting parameter)
    for i in range(len(mat.shape)):
        # build delta matrix
        # certain versions of numpy throw an "invalid value encountered" RuntimeError here but the function behaves correctly
        with np.errstate(invalid='ignore'):
            deltas_here = np.absolute(np.diff(mat,axis=i))
        pad_widths = [(0,0) for j in range(len(mat.shape))]
        pad_widths[i] = (1,1)
        deltas_here = np.pad(deltas_here, pad_widths, mode='constant', constant_values=0)

        # build "winner" matrix in this direction (ignore nans)
        # this is really ugly because we have to index in at variable positions...
        # likewise here with the error
        with np.errstate(invalid='ignore'):
            winners[i]=np.fmax(deltas_here[tuple([Ellipsis]+[slice(None,mat.shape[i],None)]+[slice(None)]*(len(mat.shape)-i-1))],deltas_here[tuple([Ellipsis]+[slice(1,mat.shape[i]+1,None)]+[slice(None)]*(len(mat.shape)-i-1))])

    grad = np.amax(winners,axis=0)

    # save these values to the appropriate indices in the vector
    if is_grid:
        return (grp, grad.flatten())
    else:
        # pick out only the boxes that exist
        return (grp, grad[tuple([i for i in list([ind_lists[p] for p in fit_param_names])])])
