#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pickle
from tools.json_file_writer import dict2json
from spatial_subsampling import get_all_grid_points
from tools.npy_file_writer import num2string
from tools.datetime_formatter import DateTimeFormatter
    
#%% Functions
def measurement_data_params(depth_all, Ndef_all, n_dtsize, starting_points, phi_all, Pndt_all):
    rng = np.random.default_rng()
    mdata_params = []
    for depth in depth_all:
#        # Shallow
#        if depth == depth_all[0]:
#            dtsize = n_dtsize[0]
#            starting_points_set = starting_points[0] # For dataNo
#        # Deep
#        else:
#            dtsize = n_dtsize[1]
#            starting_points_set = starting_points[1] # For dataNo
        for idx, Ndef in enumerate(Ndef_all):
            curr_starting_point = starting_points[idx]
            for dataNo in range(n_dtsize[idx]):
                dataNo = int(dataNo + curr_starting_point) 
                phi = rng.choice(phi_all)
                Pndt = rng.choice(Pndt_all)
                mdata_params.append(np.array([depth, Ndef, dataNo, phi, Pndt]))
                
    return mdata_params



def single_dinto_dict(depth, Ndef, dataNo, p_batch, phi, f_bin, ss_method, ss_cov, Pndt):
    dinfo = {
            'depth' : int(depth),
            'Ndef' : int(Ndef),
            'dataNo' : int(dataNo),
            'p_batch' : p_batch,
            'phi' : int(phi),
            'f_bin' : int(f_bin),
            'ss_method' : ss_method,
            'ss_cov' : ss_cov,
            'Pndt' : Pndt
            }
    return dinfo


def structure_fv_parameters(f_bin_selected, p_rows_selected, n_ss):
    """ 
    Vector structure:
        vec_f :
            Ascending order for both p_rows & spatial subsampling
            -> f0, f1, f2 ....... f24 | f0, f1, f2 ....... f24 | f0, f1, f2 ....... f24 | ....
                (p_row[0], n_ss = 0)    (p_row[0], n_ss = 1)     (p_row[0], n_ss = 2) ....
        vec_p : 
            Remains same for a set of both f_bins & spatial subsampling
            -> p_row[0], p_row[0], ..., p_row[0], | p_row[0] .....  p_row[0], | ....
                ([f0, f24], n_ss = 0)               ([f0, f24], n_ss = 1)
        vec_s :
            Remains same only for f_bin
    
    Parameters
    ----------
        f_bin_selected : np.ndarray(n_freq)
            Selected frequency bins
        p_rows_selected : np.ndarray(n_batch)
            Selected rows of p_batch_all
        n_ss : int
            Total number of samples with different spatial sampling
    """
    ss, ff, pp = np.meshgrid(np.arange(n_ss), f_bin_selected, p_rows_selected)
    vec_f = ff.flatten('F')
    vec_prow = pp.flatten('F')
    vec_ssidx = ss.flatten('F')
    
    return vec_f, vec_prow, vec_ssidx


#%% Variable setting

""" 
Clean data parametes: -> 240 varieties
    * depth
    * Ndef
    * dataNo

Data modification parameters:
    * phi (= phase)
    * noise power / dt
    
Output (FV) parameters: -> 125 varieties
    * p_batch (= batch starting point)
    * f (= freq. bin)

Input (subsampled data) parameters: -> 20 varieties
    * sampling coverage (or positions)
"""

# Dataset type
ds_type = 'test' # 'train', 'test' 'vali'

# Fixed parameters
N_batch = 10
Nx, Ny = 30, 30

# Data relevant (w.r.t. clean data or modification)
depth_all = np.array([647, 903]).astype(int) # 1159, 135, 903, 391, 647
Ndef_all = np.array([2, 5, 10]).astype(int) # 2, 5, 10
phi_all = np.arange(0, 360, 45).astype(int) # [degree]
Pndt_all = 10**-5* np.array([1]) # Noise power / dt 

### DNN output (= FV) relevant ###
# Batch starting positions: evry 5 grids
p_batch_all = 5* get_all_grid_points(int((Nx - N_batch)/5) + 1, int((Ny - N_batch)/5) + 1) # Batch starting points
# Freq. bins: bin [18, 25] = relevant bins (= must be included) 
f_all = np.arange(51) # freq. bin
f_mustincl = np.arange(15, 28)
f_optional = np.delete(f_all, f_mustincl)

### DNN input (spatial subsampling) relevant ###
ss_method_all = ['uniform', 'rndwalk']
#!!!!!! For training !!!!!!
if ds_type == 'train':
    print('Training set!')
    sscov_uniform = np.array([1, 0.95, 0.95, 0.95, 0.9, 0.9, 0.9, 0.5, 0.5, 0.4, 0.4, 0.4, 0.3, 0.3, 0.3]) 
    sscov_rndwlak = np.array([0.7, 0.9, 1.2, 1.6]) 
#!!!!! For validation/test !!!!!!
else:
    print('Validation or test set!')
    sscov_uniform = np.repeat(np.array([0.8, 0.65, 0.5, 0.4, 0.3]), 3) 
    sscov_rndwlak = np.array([0.7, 0.9, 1.2, 1.6]) 

### Number of samples ###
#n_dtsize_shallow = np.array([20, 20, 10])# for depth = 391
n_dtsize_bottom = np.array([5, 5, 5])# for depth = 647, 903
n_dtsize = n_dtsize_bottom#list([n_dtsize_shallow, n_dtsize_bottom])
n_batch = 10 
n_freq_opt = 7 # random freq. bins
n_freq = n_freq_opt + len(f_mustincl)
n_ss_single = 15 # samples for single SS method

###  Starting points to choose data from ### 
#starting_points_shallow = np.array([50, 30, 20])
starting_points_deep = np.array([25, 25, 25])
starting_points =  starting_points_deep#list([starting_points_shallow, starting_points_deep])



# Set dinfo_list with the clean data parameters
# Parameters to be included = depth, Ndef, dataNo, phase, Pndt
mdata_params = measurement_data_params(depth_all, Ndef_all, n_dtsize, starting_points, phi_all, Pndt_all)


# Complete d_info_all = nested dict composed of 625000 single dict
dinfo_all = {}

# Initialize default_rng() class
rng = np.random.default_rng()

# Set values for the rest of the parameters: 
for paramNo, param in enumerate(mdata_params):
    depth, Ndef, dataNo, phi, Pndt = param
    
    # Select values for batch positions 
    p_rows_selected = rng.choice(np.arange(p_batch_all.shape[0]), n_batch, replace = False)
    # Select values for freq. bins: must be included = 15:28 bins 
    f_bin_selected = np.sort(np.concatenate((f_mustincl, rng.choice(f_optional, n_freq_opt, replace = False)))) # shape = 20
    
    # Select valus for spatial subsampling
    # (a) Unifrom distribution = predefined
    sscov_uni_selected = np.copy(sscov_uniform)
    # (b) Random wlak 
    sscov_rw_selected = rng.choice(sscov_rndwlak, n_ss_single)
    sscov_selected = np.concatenate((sscov_uni_selected, sscov_rw_selected))
    ss_method_idx = np.repeat(np.arange(2), n_ss_single)
    
    # Vectorize
    vec_f, vec_prow, vec_ssidx = structure_fv_parameters(f_bin_selected, p_rows_selected, 2* n_ss_single)
    
    
    def update_dinfo_all(idx):
        idx = int(idx)
        # Unroll: f_bin, p_batch, ss_cov, ss_method
        f_bin = vec_f[idx]
        p_batch = p_batch_all[vec_prow[idx], :]
        ss_cov = sscov_selected[vec_ssidx[idx]]
        ss_method = ss_method_all[ss_method_idx[vec_ssidx[idx]]]
        
        # Single dinfo dictionary
        dinfo_single = single_dinto_dict(depth, Ndef, dataNo, p_batch, phi, f_bin, ss_method, ss_cov, Pndt)
        # Add the current single dictionary 
        counter = paramNo* len(vec_f) + idx
        print(counter)
        dinfo_all.update({
                str(int(counter)) : dinfo_single
                })
        
    np.apply_along_axis(update_dinfo_all, 0, np.array([np.arange(len(vec_f))]))
                
            

# Save the ML parameter dictionary
dtf = DateTimeFormatter()
today = dtf.get_date_str()
with open('params/ml_data_params_{}_{}.pickle'.format(today, ds_type), 'wb') as handle:
    pickle.dump(dinfo_all, handle, protocol = pickle.HIGHEST_PROTOCOL)        
                



