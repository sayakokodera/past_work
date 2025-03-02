#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
import time
import pickle

from tools.display_time import display_time
from tools.npy_file_writer import save_data
from tools.npy_file_writer import num2string
from tools.datetime_formatter import DateTimeFormatter

from phase_shifter import PhaseShifter
from frequency_variogram import FrequencyVariogramRaw
from spatial_subsampling import get_all_grid_points


#%% Functions
def load_and_modify_data(ml_param):
    print(' ')
    print('*** A new set of clean data is loaded & modified! ***')
    
    # Unroll ML params
    depth = ml_param['depth'] # = Nt_offset
    Ndef = ml_param['Ndef']
    dataNo = ml_param['dataNo']
    phi = ml_param['phi'] #[degree]
    Pndt = 10**-5 # Mean instantaneous noise power, fixed -> which is almost clean
    
    # Load data
    path = 'npy_data/ML/train/clean_data/depth_{}/Ndef_{}'.format(depth, num2string(Ndef))
    fname = '{}.npy'.format(num2string(dataNo))
    data_clean = np.load('{}/{}'.format(path, fname)) # shape = M x Nx x Ny
    
    # Phase shift
    ps = PhaseShifter()
    data_ps = ps.shift_phase(data_clean, phi, axis = 0) # shape = M x Nx x Ny
    
    # Add noise (very very small though)
    rng = np.random.default_rng()
    noise = rng.normal(scale = np.sqrt(Pndt), size = data_ps.shape) # shape = M x Nx x Ny
    data_roi = data_ps + noise
    
    # Free up the memory
    del data_clean, data_ps
    
    return data_roi


def compute_fv_single_batch(data_roi, p_batch, N_batch, dx, maxlag, f_range):
    print(' ')
    print('*** FV is computed! ***')
    print(' ')
    
    # Batch starting positions
    x, y = p_batch
    
    # Batch data
    data_batch = data_roi[:, x : x + N_batch, y : y + N_batch]
    data_batch = data_batch / np.abs(data_batch).max() # Normalize
    
    # Compute FV for full sampling
    s_full = np.around(dx* get_all_grid_points(N_batch, N_batch), 6) #[m]
    cfv = FrequencyVariogramRaw(grid_spacing = dx, maxlag = maxlag)
    cfv.set_positions(s_full)
    cfv.set_data(data_batch, M, f_range = f_range)
    cfv.compute_fv()
    cfv.smooth_fv(deg, ret_normalized = True)
    fv_norm = cfv.get_fv()
    fvmax = np.copy(cfv.fvmax)
    
    return fv_norm, fvmax



#%% Data & FV parameters
# ROI
Nx = 30
Ny = 30
M = 512
N_batch = 10

# Grid spacing
dx = round(0.5* 10**-3, 4) #[mm], to make sure there is no ambiguity of 10**-16 or so

# FV params
maxlag = 0.5* np.sqrt(2)* N_batch* dx # [m], fluctuation will be added in the FV class
deg = 5 # degree if fitting polynomials
f_range = np.array([0, 51]) # freq[50] = 7.8125 MHz, f_ny_max = (3.36* 1.1)* 2 = 7.392 MHz


#%% ML parameters

# Dataset relevant
""" For dataset w/ fdate = '210503_train'
Clean data parametes: -> 120 varieties
    * depth
    * Ndef
    * dataNo

Data modification parameters:
    * phi (= phase)
    * noise power / dt
    
Output (FV) parameters: -> n_batch* n_freq = 150 varieties
    * p_batch (= batch starting point)
    * f (= freq. bin)

Input (subsampled data) parameters: -> 2* n_ss_single = 30 varieties
    * sampling coverage (or positions)
    
ML parameter dictionary:
    keys = ['depth', 'Ndef', 'dataNo', 'p_batch', 'phi', 'f_bin', 'ss_method', 'ss_cov', 'Pndt']
"""
# Dataset type
ds_type = 'test' # 'train' 'test'

# From parameter dictionary setting
depth_all = np.array([647, 903])
Ndef_all = np.array([2, 5, 10])
#n_dtsize_bottom = np.array([20, 20, 20])# -> for 210503_train
n_dtsize_bottom = np.array([5, 5, 5])# -> for 210503_vali/test
n_dtsize = n_dtsize_bottom#list([n_dtsize_shallow, n_dtsize_bottom])
n_batch = 10 
n_freq_opt = 7 # random freq. bins
n_freq = n_freq_opt + len(np.arange(15, 28))
n_ss_single = 15 # samples for single SS method
# Resultiong variations
n_data = len(depth_all)* np.sum(n_dtsize_bottom) # Variations w.r.t. the clean data
n_output = n_batch* n_freq# Variations w.r.t. the output data (= FV)
n_input =  2* n_ss_single# Variations w.r.t. the input data (= spatial subsampling)

# Intervals 
itv_measdata = n_output* n_input # for loading a new clean data
itv_batch = n_freq* n_input # for a new batch = loading a new FV


# Load ML parameter dictionray
fdate = '210503_{}'.format(ds_type)
with open('params/ml_data_params_{}.pickle'.format(fdate), 'rb') as handle:
    ml_param_all = pickle.load(handle)
    
# For save the computed true FV
path_fv_norm = 'npy_data/ML/{}/outputs/fv_norm'.format(ds_type)
path_fvmax = 'npy_data/ML/{}/outputs/fvmax'.format(ds_type)

#import sys
#sys.exit()

#%% Output generation
start_all = time.time()

# Iterate over all ML parameter sets
for setNo in range(len(ml_param_all)):
    print('#===========================================#')
    print('setNo = {} / {}'.format(setNo, len(ml_param_all)))
    start = time.time()
    # Current parmeter set
    ml_param = ml_param_all[str(setNo)]
    
    # Load & modify data
    if (setNo % itv_measdata) == 0:
        data_roi = load_and_modify_data(ml_param)
        
    # Compute the true FV for the current batch
    if (setNo % itv_batch) == 0:
        fv_norm, fvmax = compute_fv_single_batch(data_roi, ml_param['p_batch'], N_batch, dx, maxlag, f_range)    
    
    # Choose the feature mapped input of the current freq. bin
    f_bin = ml_param['f_bin'] 
    data2save_fv_norm = fv_norm[f_bin, :] # shape = Nh
    data2save_fvmax = fvmax[f_bin]
    print('data2save(fv_norm) with shape = {}'.format(data2save_fv_norm.shape))
    print('data2save(fv_max) = {}'.format(data2save_fvmax))
    
    print('Current params:')
    print('depth = {}, Ndef = {}, dataNo = {}, phi = {}'.format(ml_param['depth'], ml_param['Ndef'], 
                                                                ml_param['dataNo'], ml_param['phi']))
    print('p_batch = {}'.format(ml_param['p_batch']))
    print('f_bin = {}'.format(f_bin))
    
    # Save
    fname = 'setNo{}.npy'.format(int(setNo))
    save_data(data2save_fv_norm, path_fv_norm, fname)
    save_data(data2save_fvmax, path_fvmax, fname)
    
    # For testing the script
    if (setNo % itv_batch) == 0:
        print('##########################################################')
        print('Output generation per {} sets:'.format(itv_batch))
        display_time(round(time.time() - start, 3))
        print('##########################################################')
    
#    # For test
#    if setNo == 0:
#        raise ValueError('Stop!')

    

# For printing the current time
dtf = DateTimeFormatter()
now = dtf.get_time_str()

print('##########################################################')
print('End of FV calculation')
display_time(round(time.time() - start_all, 3))
print('with current time = {}'.format(now))
print('##########################################################')



