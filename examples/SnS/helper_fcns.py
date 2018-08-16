from copy import deepcopy
import pandas as pd
import deepdish as dd
import numpy as np
import zipfile

def get_new_data(model_obj, new_pts_path, new_data_path):
    all_data_path = './sim_data_less_sparse.h5'

    # unzip the larger data file
    zip_ref = zipfile.ZipFile(all_data_path+'.zip', 'r')
    zip_ref.extractall('.')
    zip_ref.close()

    new_pts = dd.io.load(new_pts_path)
    all_data = dd.io.load(all_data_path)
    new_grps = new_pts.groupby(model_obj.fit_param_names())
    all_grps = all_data.groupby(model_obj.fit_param_names())

    new_dEc = sorted(list(set(new_pts['dEc'])))
    new_S_eff = sorted(list(set(new_pts['S_eff'])))
    new_mu = sorted(list(set(new_pts['mu'])))
    new_tau = sorted(list(set(new_pts['tau'])))
    new_lists = [new_dEc, new_S_eff, new_mu, new_tau]

    all_dEc = sorted(list(set(all_data['dEc'])))
    all_S_eff = sorted(list(set(all_data['S_eff'])))
    all_mu = sorted(list(set(all_data['mu'])))
    all_tau = sorted(list(set(all_data['tau'])))
    old_lists = [all_dEc, all_S_eff, all_mu, all_tau]

    # map new values to old indices
    ind_maps = [{},{},{},{}]
    for i in range(len(new_lists)):
        for new_val in new_lists[i]:
            diffs = [abs(new_val-old_val) for old_val in old_lists[i]]
            min_ind = diffs.index(min(diffs))
            ind_maps[i][new_val] = min_ind
        #diffs = np.diff(list(ind_maps[i].values()))
        #if any([d==1 for d in diffs]):
        #    print('Might be out of range!'+str(ind_maps[i]))

    # map new values onto unrounded sim points
    key_map = {}
    for new_key in list(new_grps.groups.keys()):
        old_key = [-1, -1, -1, -1]
        for i in range(len(model_obj.params.fit_params)):
            old_key[i] = old_lists[i][ind_maps[i][new_key[i]]]
        key_map[new_key] = tuple(old_key)
    keys_to_keep = key_map.values()

    new_data = deepcopy(all_data)
    inds_to_drop = []
    for key in list(all_grps.groups.keys()):
        if not key in keys_to_keep:
            inds_to_drop.extend(list(all_grps.groups[key]))

    new_data.drop(inds_to_drop, inplace=True)
    new_data.reset_index(drop=True, inplace=True)
    dd.io.save(new_data_path, new_data)
