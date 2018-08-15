from copy import deepcopy
import pandas as pd
import deepdish as dd
import numpy as np

def get_new_data(model_obj, all_data_path, new_pts_path, new_data_path):
    new_pts = dd.io.load(new_pts_path)
    all_data = dd.io.load(all_data_path)
    new_grps = new_pts.groupby(model_obj.fit_param_names())
    all_grps = all_data.groupby(model_obj.fit_param_names())

    new_EA = sorted(list(set(new_pts['EA'])))
    new_Nt_i = sorted(list(set(new_pts['Nt_i'])))
    new_mu = sorted(list(set(new_pts['mu'])))
    new_Nt_SnS = sorted(list(set(new_pts['Nt_SnS'])))
    new_lists = [new_EA, new_Nt_i, new_mu, new_Nt_SnS]
    
    all_EA = sorted(list(set(all_data['EA'])))
    all_Nt_i = sorted(list(set(all_data['Nt_i'])))
    all_mu = sorted(list(set(all_data['mu'])))
    all_Nt_SnS = sorted(list(set(all_data['Nt_SnS'])))
    old_lists = [all_EA, all_Nt_i, all_mu, all_Nt_SnS]
    
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
