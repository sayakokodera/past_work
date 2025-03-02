#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML datageneration: input with scalar-valued lags
"""
import numpy as np
import time
import pickle
import matplotlib.pyplot as plt

from tools.display_time import display_time
from tools.npy_file_writer import save_data
from tools.npy_file_writer import num2string
from tools.datetime_formatter import DateTimeFormatter

from spatial_subsampling import batch_subsampling
from spatial_subsampling import batch_random_walk2D
from phase_shifter import PhaseShifter
from frequency_variogram import FrequencyVariogramDNN
from frequency_variogram import FrequencyVariogramRaw


start_all = time.time()

#%% Functions

def load_and_modify_data(ml_param):
    # Unroll ML params
    depth = ml_param['depth'] # = Nt_offset
    Ndef = ml_param['Ndef']
    dataNo = ml_param['dataNo']
    phi = ml_param['phi'] #[degree]
    Pndt = ml_param['Pndt'] # noise power / dt
    
    # Load data
    path = 'npy_data/ML/train/clean_data/depth_{}/Ndef_{}'.format(depth, num2string(Ndef))
    fname = '{}.npy'.format(num2string(dataNo))
    data_clean = np.load('{}/{}'.format(path, fname)) # shape = M x Nx x Ny
    
    # Phase shift
    ps = PhaseShifter()
    data_ps = ps.shift_phase(data_clean, phi, axis = 0) # shape = M x Nx x Ny
    
    # Add noise
    rng = np.random.default_rng()
    noise = rng.normal(scale = np.sqrt(Pndt), size = data_ps.shape) # shape = M x Nx x Ny
    data_roi = data_ps + noise
    
    # Free up the memory
    del data_clean, data_ps
    
    return data_roi


def data_sampling_and_feature_mapping(ml_param, data_roi, dx, N_batch, maxlag, l_cnnwin, smooth = False,  deg = None):
    print(' ')
    print('*** Spatial subsampling & feature mapping ***')
    # Choose the current batch
    x, y = ml_param['p_batch']
    data_batch = data_roi[:, x : x + N_batch, y : y + N_batch]
    
    # Spatial subsampling
    N_scan = int(ml_param['ss_cov']* N_batch**2)
    # Initial setting
    s_smp = np.zeros((1, 2))
    ss_counter = 0
    # In case s_smp contains less than 30 points -> select another set of positions
    while s_smp.shape[0] < 0.3* N_batch**2:
        print('Spatial Subsampling: tiral No.{}'.format(ss_counter))
        if ml_param['ss_method'] == 'uniform':
            s_smp = batch_subsampling(N_batch, N_scan) # UNITLESS
        else: # random walk
            s_smp = batch_random_walk2D(N_scan, N_batch) # UNITLESS
    print('Real coverage after subsampling = {} / 100'.format(s_smp.shape[0]))
    # Pick A-Scans
    data_smp = data_batch[:, s_smp[:, 0], s_smp[:, 1]]
    
    # Histogram
    cfv = FrequencyVariogramDNN(dx, N_batch, maxlag = maxlag)
    cfv.set_positions(np.around(dx* s_smp, 10)) #Don't forget ro multiply with dx!!!!!!! 
    cfv.set_data(data_smp, M, f_range = f_range)
    hist = cfv.compute_histogram_scalarlags() # hist.shape = 25
    lags_full = cfv.get_features()
    
    # Normalized FV
    if smooth == False:
        fv = cfv.feature_mapping_scalar_valued() # fv.shape = Nf x 25, 
        fvmax = cfv.get_fvmax() # shape = Nf
    
    # Using smoothed raw FV
    else:
        print('Smooth FV!')
        del cfv
        cfv = FrequencyVariogramRaw(grid_spacing = dx, maxlag = maxlag)
        cfv.set_positions(np.around(dx* s_smp, 10)) #Don't forget ro multiply with dx!!!!!!! 
        cfv.set_data(data_smp, M, f_range = f_range)
        cfv.compute_fv()
        cfv.smooth_fv(deg, ret_normalized = True, lags_fv = lags_full)
        fv = cfv.get_fv() # fv.shape = Nf x 25
        fvmax = cfv.get_fvmax() # shape = Nf
        print('fv.shape = {}'.format(fv.shape))
        
        # Check the dimension
        if fv.shape[1] != len(lags_full):
            raise ValueError('Compute FV: smoothed FV has the different size from the full lags!')
    
    # Extend the frequency bins in the negative direction
    fv = extend_frequency_bins(fv, l_cnnwin)
    fvmax = extend_frequency_bins(fvmax, l_cnnwin)
    
    print(' ')
    
    return fv, hist, fvmax
    

def extend_frequency_bins(fv_feat, l_cnnwin):
    """ Extend freq. bins by int(l_cnnwin/2) in the negative freq. components
    (e.g.) l_cnnwin = 5 => extensions by 2 bins
    """
    # Freq. responses to add (in the negative freq. direction)
    fv2add = fv_feat[1:int(l_cnnwin/2) + 1]
    # Flip 
    fv2add = np.flip(fv2add, axis = 0)
    # Extend
    fv_feat_extended = np.concatenate((fv2add, fv_feat), axis = 0)
    
    return fv_feat_extended 
    
    
    
    

#%% Data & FV parameters
# ROI
Nx = 30
Ny = 30
M = 512
N_batch = 10

# Grid spacing
dx = 0.5* 10**-3 #[m], safe for dy

# FV params
maxlag = 0.5* np.sqrt(2)* N_batch* dx # [m], add fluctuation!!!!!

# FV types: smoothed or non-smoothed
smooth = True #False
deg = 5 # None


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
    
Output (FV) parameters: -> n_batch* n_freq = 200 varieties
    * p_batch (= batch starting point)
    * f (= freq. bin)

Input (subsampled data) parameters: -> 2* n_ss_single = 30 varieties
    * sampling coverage (or positions)
    
ML parameter dictionary:
    keys = ['depth', 'Ndef', 'dataNo', 'p_batch', 'phi', 'f_bin', 'ss_method', 'ss_cov', 'Pndt']
"""
# Dataset type
ds_type = 'train' # 'train', 'test', 'vali'

# From parameter dictionary setting
depth_all = np.array([647, 903])
Ndef_all = np.array([2, 5, 10])
#n_dtsize_bottom = np.array([20, 20, 20])# -> for 210503_train
n_batch = 10 
n_freq_opt = 7 # random freq. bins
n_freq = n_freq_opt + len(np.arange(15, 28))
n_ss_single = 15 # samples for single SS method
# Resultiong variations
n_output = n_batch* n_freq# Variations w.r.t. the output data (= FV)
n_input =  2* n_ss_single# Variations w.r.t. the input data (= spatial subsampling)


# Intervals 
itv_measdata = n_output* n_input # for loading a new clean data
itv_ss = n_freq # for a new spatial subsampling

# CNN parameter: window size (= how many f_bins are fed)
l_cnnwin = 3 # -> f_range extension by int(l_cnnwin/2)

# Frequency ranges for FFT to save memory (with extension for using inputs in CNN)
f_range = np.array([0, 51 + int(l_cnnwin/2)]) #corresponds to np.arange(51) in param set 


# Load ML parameter dictionray
fdate = '210511_{}'.format(ds_type)
with open('params/ml_data_params_{}.pickle'.format(fdate), 'rb') as handle:
    ml_param_all = pickle.load(handle)
    
# For save the feature-mapped inputs
path_fv_norm = 'npy_data/ML/{}/input_scalarlags_smooth/fv_norm'.format(ds_type)
path_fvmax = 'npy_data/ML/{}/input_scalarlags_smooth/fvmax'.format(ds_type)
path_hist = 'npy_data/ML/{}/input_scalarlags_smooth/hist'.format(ds_type)

  
#import sys
#sys.exit()  
#%% Compute inputs
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
        
    # Spatial subsampling & feature mapping
    #!!! here the FVs are extended, shape = (Nf + l_cnnwin -1) x Nh !!!!!!!
    if (setNo % itv_ss) == 0:
        fv_norm, hist, fvmax = data_sampling_and_feature_mapping(ml_param, data_roi, dx, N_batch, maxlag, l_cnnwin,
                                                                 smooth = smooth, deg = deg)
        
    # Choose the feature mapped input of the current freq. bin
    f_bin = ml_param['f_bin'] 
    # Adjusted to the extended bins
    f_offset = int(l_cnnwin/2)
    f_windows = np.arange(f_bin - f_offset, f_bin + f_offset + 1) + f_offset
    
    data2save_fvnorm = fv_norm[f_windows, :] # shape = l_cnnwin x Nlag
    data2save_fvmax = fvmax[f_windows] # shape = l_cnnwin x 1
    data2save_hist = np.copy(hist)
    
    print('data2save_fvnorm.shape = {}'.format(data2save_fvnorm.shape))
    
    print('Current params:')
    print('depth = {}, Ndef = {}, dataNo = {}'.format(ml_param['depth'], ml_param['Ndef'], ml_param['dataNo']))
    print('p_batch = {}, f_bin = {}'.format(ml_param['p_batch'], f_bin))
    print('f_offset = {}, freq. bin window = {}'.format(f_offset, f_windows))
    print('ss_method = {}, ss_cov = {}, Pndt = {}'.format(ml_param['ss_method'], ml_param['ss_cov'], ml_param['Pndt']))
    
    # Save
    fname = 'setNo{}.npy'.format(setNo)
    save_data(data2save_fvnorm, path_fv_norm, fname)
    save_data(data2save_fvmax, path_fvmax, fname)
    save_data(data2save_hist, path_hist, fname)
    
    display_time(round(time.time() - start, 3))
    
    # For testing the script
#    if setNo == 0:
#        raise ValueError('Stop!')

    

# For printing the current time
dtf = DateTimeFormatter()
now = dtf.get_time_str()

print('##########################################################')
print('End of inputs generation')
display_time(round(time.time() - start_all, 3))
print('with current time = {}'.format(now))
print('##########################################################')



