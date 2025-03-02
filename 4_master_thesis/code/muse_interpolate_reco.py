#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Subsampled ;USE data SAFT reco

"""

import numpy as np
import matplotlib.pyplot as plt
import json
import scipy.io as scio
from scipy import ndimage
import time
import scipy.spatial.distance as scdis

from tensorflow import keras


from ultrasonic_imaging_python.forward_models.propagation_models_3D_single_medium import PropagationModel3DSingleMedium
from ultrasonic_imaging_python.reconstruction.saft_grided_algorithms import SAFTEngine
from ultrasonic_imaging_python.utils.progress_bars import TextProgressBarFast
from ultrasonic_imaging_python.visualization.slice_figures import SliceFigure3D
from ultrasonic_imaging_python.definitions import units
ureg = units.ureg

from tools.display_time import display_time
from tools.npy_file_writer import save_data
from tools.datetime_formatter import DateTimeFormatter

from smart_inspect_data_formatter import SmartInspectDataFormatter
from spatial_subsampling import batch_subsampling, batch_random_walk2D, get_all_grid_points
from array_access import search_vector

from frequency_variogram import FrequencyVariogramDNN
from frequency_kriging import FrequencyKrigingBlock
from general_batch_interpolation import InverseDistanceWeighting, moving_window_averaging


plt.close('all') 

# %% Functions
def get_opening_angle(c0, D, fC):
    r""" The opening angle (= beam spread), theta [grad], of a trnsducer element can be calculated with
        np.sin(theta) = 1.2* c0 / (D* fC) with
            c0: speed of sound [m/S]
            D: element diameter [m]
            fC: career frequency [Hz]
    (Cf: https://www.nde-ed.org/EducationResources/CommunityCollege/Ultrasonics/EquipmentTrans/beamspread.htm#:~:text=Beam%20spread%20is%20a%20measure,is%20twice%20the%20beam%20divergence.)
    """
    theta_rad = np.arcsin(1.2* c0/ (D* fC))
    return np.rad2deg(theta_rad)  


def convert2cscan(data_3d):
    cscan = np.max(np.abs(data_3d), axis = 0) 
    return cscan


def plot_cscan(data, title, dx, dy, vmin = None, vmax = None):
    # !!!!!!! Swap axes to align to the MUSE CAD image !!!!!
    cscan = np.swapaxes(data, 0, 1)
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cscan, vmin = vmin, vmax = vmax)#, extent=[xrange[0], xrange[-1], yrange[0], yrange[-1]])
    ax.set_aspect(dx/dy)
    ax.set_title(title)
    ax.set(xlabel = 'x / dx', ylabel = 'y / dy')
    ax.invert_yaxis()
    plt.colorbar(im)
    
    del fig
    
def plot_bscan(data_slice, title, dz, dx, vmin = None, vmax = None):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(data_slice, vmin = vmin, vmax = vmax)#, extent=[xrange[0], xrange[-1], yrange[0], yrange[-1]])
    ax.set_aspect(dx/dz)
    ax.set_title(title)
    ax.set(xlabel = 'x / dx', ylabel = 'z / dz')
    plt.colorbar(im)
    
    del fig
    

def calc_reco(raw_data, saft_engine, save = False, dict_f = None):
    print('#======= SAFTing =======#')
    start = time.time()
    
    # Adjust the data structure
    # SAFTEngine is built for multi-channel phased array measurements
    # -> here: single channel, pulse-echo measurement
    # => just add new (empty) dimension to match the data structure
    data = np.copy(raw_data)
    data = data[:,  np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, :]
    
    # Progress bar: initializazion
    progress_bar = TextProgressBarFast(10, 'SAFT progress', 1, 1)
    # Reco
    reco = saft_engine.get_reconstruction(data, progress_bar=progress_bar)
    
    # Progress bar: finalization
    progress_bar.finalize()
    
    display_time(round(time.time() - start, 3))
    
    if save == True:
        save_data(reco, dict_f['path'], dict_f['fname'])
        del data, reco
    else:
        del data
        return reco


def gaussian_smoothing(arr, M, sigma = 5):
    arrsmt = np.zeros(arr.shape)
    for row in range(M):
        arrsmt[row, :, :] = ndimage.gaussian_filter(arr[row, :, :], sigma = 5)
    
    return arrsmt


def rotate_coordinate(angle_deg, pos):
    theta = np.deg2rad(angle_deg)
    mat = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
            ])
    pos_rot = np.matmul(mat, pos[..., np.newaxis])[:, :, 0]
    print(pos_rot.shape)
    return pos_rot
 

def ss_randomwalk(Ns, Nx, Ny, seedNo):
    N_walk = 0
    seed_counter = 0
    while N_walk < Ns:
        print('Spatial subsampling! No. {}'.format(seed_counter))
        walk = batch_random_walk2D(3*Ns, Nx, Ny, seed = seedNo + seed_counter)
        #walk = batch_random_walk2D(20* Ns, Nx, Ny, seed = seedNo + seed_counter)
        N_walk = walk.shape[0]
        # Select Ns positions from the walk
        if N_walk >= Ns:
            rng = np.random.default_rng()
            indices = rng.choice(np.arange(N_walk), Ns, replace = False, shuffle = True) # replace == False -> no repeats!
            walk_Ns = walk[indices, :]
        
        seed_counter += 1 
    
    return walk_Ns



def batch_moving_window_assignment(batch_data, data_roi, x_start, y_start):
    # Dimension
    N_batch = batch_data.shape[1]
    Nx = data_roi.shape[1]
    Ny = data_roi.shape[2]
    
    # Copy
    out = np.copy(data_roi)
    oldbatch = np.copy(out[:, x_start : x_start + N_batch, y_start : y_start + N_batch])
    
    # Scaling via moving window averaging
    newbatch = moving_window_averaging(batch_data, x_start, y_start, N_batch, Nx, Ny)
    
    # Assignment
    # Case: there is no NaNs -> just add
    if np.all(np.isnan(oldbatch)) == False and np.all(np.isnan(newbatch)) == False:
        out[:, x_start : x_start + N_batch, y_start : y_start + N_batch]  += newbatch
        
    # Case: there are NaNs -> converting NaNs to 1j (easier to handle) (for RMSE matrix)
    else:
        
        def nan2imag(arr):
            out = np.ones(arr.shape, dtype = complex)
            out = out* arr
            # Positions of NaNs
            x, y = np.nonzero(np.isnan(arr) == True)
            # NaN -> 1j
            if len(x) > 0:
                out[x, y] = 1j
            return out
        
        # Replace NaNs with 1j
        old_complex = nan2imag(oldbatch[0, :, :]) # shape = N_batch x N_batch
        new_complex = nan2imag(newbatch[0, :, :]) # shape = N_batch x N_batch
        # Add
        summed = old_complex + new_complex
        # Find where both oldbatch and newbatch have NaNs (which is now represented as 2j in summed)
        x, y = np.nonzero(summed == 2*1j)
        # Replace 2j with NaN, s.t. mutual NaNs remain as NaNs
        summed[x, y] = np.nan
        
        # Assignment = real part of summed 
        out[:, x_start : x_start + N_batch, y_start : y_start + N_batch] = summed.real
    
    return out


def resampling_random(p_smp_roi, Nx, Ny, Ns_resmp, combine = True):
    ### Identify the unscanned positions ###
    # Full grid points in the ROI
    p_full_roi = get_all_grid_points(Nx, Ny)
    # Indices of the sampled positions
    idx_smp = np.zeros(p_smp_roi.shape[0], dtype = int)
    for row, p in enumerate(p_smp_roi):
        idx_smp[row] = search_vector(p, p_full_roi, 1)
    # Unscanned positions
    p_empty_roi = np.delete(p_full_roi, idx_smp, axis = 0)
    
    ### Random sampling ### 
    idx_resmp_rnd = np.random.choice(np.arange(p_empty_roi.shape[0]), Ns_resmp, replace = False) # shape = Ns
    p_resmp = p_empty_roi[idx_resmp_rnd, :] # shape = Ns x 2
    
    ### Combine with the original points ###
    if combine == True:
        return np.concatenate((p_smp_roi, p_resmp), axis = 0) # shape = 2* Ns x 2
    else:
        return p_resmp # shape = Ns x 2


def resampling_krig(p_smp_roi, p_resmp_rnd, rmse, rth, Nx, Ny, Ns_resmp, combine = True):
    """ Resampling scheme accroding to the estimated prediction error after FK (the rest are 'filled' with random 
    scheme)
    Parameters
    ----------
        p_smp_roi : np.array((Ns_org, 2))
            Initial sampled positions
        p_resmp_rnd: np.array((Ns_resmp, 2))
            Random resampling to be used to fill
        rmse : np.array((Nx, Ny)) (incl. NaN)
            Predicted RMSE map to be used to decide the resamling positions
        rth : int, float
            Relative thresold value (w.r.t. the mean(rmse)) to be used for determining the resampling positions 
        Nx, Ny: int
            Batch size
        Ns_resmp : int
            Number of the resampling positzions
    
    """
    ### (1) Resampling positions accroding to the RMSE ###
    # Identify the high inaccurate points 
    x, y = np.nonzero(rmse >= rth* np.nanmean(rmse))
    Ns_krig = len(x)
    
    # case: there are enough points to "optimally" resample
    if Ns_krig > Ns_resmp:
        # Choose N_resmp points 
        indices = np.random.choice(np.arange(len(x)), Ns_resmp, replace = False)
        p_resmp = np.array([x[indices], y[indices]]).T # shape = Ns_resmp x 2
    
    # case: need to "fill" the sampling
    else:
        p_resmp_krig = np.array([x, y]).T # shape = K x 2
        ### (2) Random resampling to "fill" the sampling ###
        # Number of "filling" points
        Ns_rnd = Ns_resmp - Ns_krig # = Ns_resmp - K
        print('Random resampling to fill: Ns_rnd = {}'.format(Ns_rnd))
        # Select Ns_rnd elements from p_resmp_rnd
        indices = np.random.choice(np.arange(p_resmp_rnd.shape[0]), Ns_rnd, replace = False) # shape = Ns_rnd
        print('indices.shape = {}'.format(indices.shape))
        
        ### (3) Comnbine ###
        p_resmp = np.concatenate((p_resmp_krig, p_resmp_rnd[indices, :]), axis = 0) # shape = Ns_resmp x 2
    
    ### Combine with the original points ###
    if combine == True:        
        return np.concatenate((p_smp_roi, p_resmp), axis = 0), Ns_krig # shape = (Ns_org + Ns_resmp) x 2
    else:
        return p_resmp, Ns_krig
    

def make_scanmap(p_smp_roi, Nx, Ny):
    scan_map = np.zeros((Nx, Ny))
    scan_map[p_smp_roi[:, 0], p_smp_roi[:, 1]] = 1
    return scan_map
    
    
    
    
    


#%% MUSE parameters 
# Variables: ROI
zmin, zmax = 1888, 2400 # = Nt_offset, Nt, 1865....1895 = only noise, 1888, 2400
xmin, xmax = 185, 235#base = 240, 350   115, 145 
ymin, ymax = 20, 70 #base = 115, 165
M = zmax - zmin
Nx = xmax - xmin
Ny = ymax - ymin

# Measurement setting
c0 = 5900 #[m/S]
fS = 80*10**6 #[Hz]  
fC = 3.36*10**6 #[Hz] 
D = 10*10**-3 #[m], transducer element diameter
opening_angle = get_opening_angle(c0, D, fC)
dx = 0.5*10**-3 #[m], dy takes also the same value
dt = 1/fS #[S]
dz = 0.5* c0* dt #[m]

# Pulse model parameters (estimated from the measurement data)
alpha = 10.6*10**12 #[Hz]**2, bandwidth factor
r = 0.75 # Chirp rate


#%% MUSE data
""" Info regarding the measurement data
* Raw data is flipped w.r.t. y-axis (compared to the CAD image)
    -> use np.flip(data, axis = 2)
* Back-wall echoes are removed
* Displaying C-Scan: use transpose, i.e. np.max(np.abs(data), axis = 0).T

"""
# Load
muse_data = scio.loadmat('MUSE/measurement_data.mat')
print(muse_data['data'].shape)
#plot_cscan(convert2cscan(muse_data['data']), 'Entire MUSE C-Scan', dx, dx)


# Select the ROI
A_roi = muse_data['data'][zmin:zmax, xmin:xmax, ymin:ymax]
A_roi = A_roi / np.abs(A_roi).max() # normalize
del muse_data['data']

plot_cscan(convert2cscan(A_roi), 'C-Scan of ROI', dx, dx)


#%% Spatial subsampling
# Parameters
Ns = int(0.08* Nx* Ny) # Number of scans, initial setting
seedNo = np.random.randint(10**5)#36953


p_smp_roi = np.load('npy_data/simulations/210625_8%/setNo_6/p_smp.npy')
rmse_fkdnn = np.load('npy_data/simulations/210625_8%/setNo_6/rmse_FK.npy')

raise ValueError('Stop!')

# Position selection
# Grid points

N_smp = 0
counter = 1
while N_smp < Ns:
    # Set 1
#    p_smp1 = ss_randomwalk(Ns = int(0.1* Ns), Nx = np.random.randint(17, 20), Ny = np.random.randint(12, 16), 
#                           seedNo = seedNo + counter)
    p_smp1 = batch_subsampling(Nx = np.random.randint(17, 20), N_scan = int(0.2* Ns), Ny = np.random.randint(12, 16), 
                               seed = seedNo+ counter)
    p_smp1[:, 0] += np.random.randint(6, 10)
    p_smp1[:, 1] += np.random.randint(25, 29)
    
    # Set2
#    p_smp2 = ss_randomwalk(Ns = int(0.1* Ns), Nx = np.random.randint(17, 20), Ny = np.random.randint(12, 16), 
#                           seedNo = seedNo + 100 + counter)
    p_smp2 = batch_subsampling(Nx = np.random.randint(17, 20), N_scan = int(0.2* Ns), Ny = np.random.randint(12, 16), 
                               seed = seedNo + 100 + counter)
    p_smp2[:, 0] += np.random.randint(27, 32)
    p_smp2[:, 1] += np.random.randint(4, 9)
    
    # "Noise"
    p_smp3 = batch_subsampling(Nx = Nx, N_scan = int(1.0* Ns), Ny = Ny, 
                               seed = seedNo + 200+ counter)
    
    # Combine
    p_smp_roi = np.concatenate((p_smp1, p_smp2, p_smp3), axis = 0)
    p_smp_roi = np.unique(p_smp_roi, axis = 0)
    
    # Check
    N_smp = p_smp_roi.shape[0]
    counter += 1
    
    # Select Ns positions from the walk
    if N_smp >= Ns:
        rng = np.random.default_rng()
        indices = rng.choice(np.arange(N_smp), Ns, replace = False, shuffle = True) # replace == False -> no repeats!
        p_smp_roi = p_smp_roi[indices, :]

    
# Convert into metre
#s_smp_roi = np.around(dx* p_smp_roi, 10) # in [m]
        
        
# Random batch sampling
#p_smp_roi = batch_subsampling(Nx = Nx, N_scan = Ns, Ny = Ny, seed = seedNo)

# Scan map
scan_map = np.zeros((Nx, Ny))
scan_map[p_smp_roi[:, 0], p_smp_roi[:, 1]] = 1
plot_cscan(scan_map, 'Scan positions', dx, dx)

# Measurement data of the selected sampling positions
formatter = SmartInspectDataFormatter(p_smp_roi, 
                                      xmin = 0, xmax = Nx, 
                                      ymin = 0, ymax = Ny)
A_smp = formatter.get_data_matrix_pixelwise(A_roi[:, p_smp_roi[:, 0], p_smp_roi[:, 1]]) # size = M x Nx x Ny

plot_cscan(convert2cscan(A_smp), 'C-Scan of A_smp', dx, dx)


#%% Interpolation
### Interpolation parameters ###
N_batch = 10
p_full = get_all_grid_points(N_batch, N_batch) # Full grid points in the batch, shape = N_batch**2 x 2, unitless
# All batch starting positions
# With overlapping
p_batch_all = int(N_batch/2)* get_all_grid_points(int((Nx - N_batch)/5) + 1, int((Ny - N_batch)/5) + 1)  
# Without overlapping
#p_batch_all = N_batch* get_all_grid_points(int(Nx/N_batch), int(Ny/N_batch))

# Max. lag range
maxlag = np.around(dx* N_batch/2* np.sqrt(2), 10) # in [m]

# Parameters for FK & FV
[fmin, fmax] = [0, 50] #[fmin, fmax], corresponds to bin (i.e. index), None if we don't want to limit the freq. range
deg = 5 # Polynomial degree for FV
tik_alpha = 0.001*10**3 # Tikhonov regularization for Kriging weights calculation+
modelname = 'tf_models/conv1d_{}/model'.format('L1reg_210512') # Model for FV DNN
l_cnnwin = 3 # Window length fo CNN 1D

#import sys
#sys.exit()

# RMSE threshold
#rmse_thrate = 3

def batch_interpolation(p_start, scan_map):
    print('=======================================')
    print('p_start = {}'.format(p_start))
    # Unfold: the batch starting point
    x_start, y_start = p_start
    
    # Scan positions in the current batch
    batch_scan_indices = np.flatnonzero(scan_map[x_start : x_start+N_batch, y_start : y_start+N_batch])
    p_smp = p_full[batch_scan_indices, :] # shape = Ns x 2, unitless
    s_smp = np.around(dx* p_smp, 10) # shape = Ns x 2, in [m]
    
    print('x_smp = {}'.format(p_smp[:, 0]))
    print('y_smp = {}'.format(p_smp[:, 1]))

    
    if p_smp.shape[0] == 0:
        Ahat_idw = np.zeros((M, N_batch, N_batch))
        Ahat_DNN = np.copy(Ahat_idw)
        Ahat_hybrid = np.copy(Ahat_idw)
        rmse_FKDNN = np.nan* np.ones((N_batch, N_batch))
    
    # Case: (i) samples < 5 or (ii) min(lags) > maxlag -> no interpolation 
    elif p_smp.shape[0] < 5 or min(scdis.pdist(s_smp)) > maxlag:
        print('there are not enough valid points!')
        x_smp, y_smp = p_smp[0, :]
        
        # Base
        Ahat_idw = np.zeros((M, N_batch, N_batch))
        rmse_FKDNN = np.nan* np.ones((N_batch, N_batch))
        
        # Assign the observed data
        Ahat_idw[:, p_smp[0, 0], p_smp[0, 1]] = A_roi[:, x_start + x_smp, y_start + y_smp]
        Ahat_DNN = np.copy(Ahat_idw)
        Ahat_hybrid = np.copy(Ahat_idw)
        rmse_FKDNN[p_smp[0, 0], p_smp[0, 1]] = 0.0
          
    else:
        # Sampled data in the current batch
        A_smp_2d = A_roi[:, x_start + p_smp[:, 0], y_start + p_smp[:, 1]] # shape = M x Ns
        
        ### Batch IDW ###
        bidw = InverseDistanceWeighting(s_smp, A_smp_2d, maxlag)
        Ahat_idw = bidw.predict_blockwise(N_batch, dx)
        
        ### Batch FK ###
        start = time.time()
        
        # FV estimation
        model = keras.models.load_model(modelname)
        cfvDNN = FrequencyVariogramDNN(dx, N_batch, maxlag = maxlag)
        cfvDNN.set_positions(s_smp) 
        cfvDNN.feature_mapping(A_smp_2d, M, l_cnnwin, deg, f_range = [fmin, fmax]) 
        cfvDNN.compute_fv(model)
        fvnorm_DNN = cfvDNN.get_fv()
        fv_DNN = cfvDNN.denormalize_fv(fvnorm_DNN)
        lags_full = cfvDNN.get_features()
        # Block FK
        bfk = FrequencyKrigingBlock(s_smp, N_batch, dx, lags_full) 
        Ahat_DNN = bfk.predict(fv_DNN, A_smp_2d, tik_alpha, f_range = [fmin, fmax]) # shape = M x N_batch x N_bacth
        rmse_FKDNN = bfk.get_prediction_error() # shape = N_batch x N_bacth
        
        # Hybrid solution
        Ahat_hybrid = np.copy(Ahat_DNN)
        x, y = np.nonzero(np.isnan(rmse_FKDNN) == True)
        Ahat_hybrid[:, x, y] = Ahat_hybrid[:, x, y]
        
        display_time(round(time.time() - start, 5))
     
    return Ahat_idw, Ahat_DNN, rmse_FKDNN, Ahat_hybrid
        
            
            
    
A_idw = np.zeros(A_roi.shape)
A_fkdnn = np.zeros(A_roi.shape)
rmse_fkdnn = np.zeros((Nx, Ny))
A_hybrid = np.zeros(A_roi.shape)

start_all = time.time()
for batchNo, p in enumerate(p_batch_all):
    # Interpolate
    arr_idw, arr_fkdnn, e_fkdnn, arr_hybrid = batch_interpolation(p, scan_map)
        
    # Assign: scaling according to moving window averaging -> assignment
    x_start, y_start = p
    A_idw = batch_moving_window_assignment(arr_idw, A_idw, x_start, y_start)
    A_fkdnn = batch_moving_window_assignment(arr_fkdnn, A_fkdnn, x_start, y_start)
    rmse_fkdnn = batch_moving_window_assignment(e_fkdnn, rmse_fkdnn[np.newaxis, ...], x_start, y_start)[0, ...]
    A_hybrid = batch_moving_window_assignment(arr_hybrid, A_hybrid, x_start, y_start)
    
    
display_time(round(time.time() - start_all, 5))


# rmse_fkdnn
plot_cscan(rmse_fkdnn, 'RMSE', dx, dx, vmin = None, vmax = None)


# Discard the high inaccurate predictions
x, y = np.nonzero(rmse_fkdnn > 3* np.nanmean(rmse_fkdnn))
A_hybrid[:, x, y] = A_idw[:, x, y]

# Data
#plot_cscan(convert2cscan(A_hybrid), 'A_hybrid', dx, dx, vmin = None, vmax = None)
#plot_cscan(convert2cscan(A_fkdnn), 'A_fkdnn', dx, dx, vmin = None, vmax = None)
#plot_cscan(convert2cscan(A_idw), 'A_idw', dx, dx, vmin = None, vmax = None)

# Smoothed data
#plot_cscan(convert2cscan(gaussian_smoothing(A_hybrid, M, sigma = 5)), 'Smoothed A_hybrid', dx, dx)
#plot_cscan(convert2cscan(gaussian_smoothing(A_fkdnn, M, sigma = 5)), 'Smoothed A_fkdnn', dx, dx)
#plot_cscan(convert2cscan(gaussian_smoothing(A_idw, M, sigma = 5)), 'Smoothed A_idw', dx, dx)


#%% Resampling
# (a) random resampling (b) "optimal" resampling
# Parameter
rth_resmpkrig = 2

### Resampling positions ###
# (a) Random
p_resmp_rnd = resampling_random(p_smp_roi, Nx, Ny, Ns)
scanmap_rernd = make_scanmap(p_resmp_rnd, Nx, Ny)
# (b) "Optimal"
p_resmp_krig, Ns_krig = resampling_krig(p_smp_roi, p_resmp_rnd, rmse_fkdnn, rth_resmpkrig, Nx, Ny, Ns)
scanmap_rekrig = make_scanmap(p_resmp_krig, Nx, Ny)

# Base
A_re_idw = np.zeros(A_roi.shape)
A_re_fk = np.zeros(A_roi.shape)
rmse_re_fk = np.zeros((Nx, Ny))

raise ValueError('Stop!')


### Inteprolation ###
for batchNo, p in enumerate(p_batch_all):
    # Interpolation
    # (a) Random resampling
    arr_idw, _, _, _ = batch_interpolation(p, scanmap_rernd)
    print('Done with itp for random resampling!')
    # (b) "Optimal" resampling
    _, arr_fkdnn, e_fkdnn, _ = batch_interpolation(p, scanmap_rekrig)
    print('Done with itp for Kriging resampling!')
        
    # Assign: scaling according to moving window averaging -> assignment
    x_start, y_start = p
    A_re_idw = batch_moving_window_assignment(arr_idw, A_re_idw, x_start, y_start)
    A_re_fk = batch_moving_window_assignment(arr_fkdnn, A_re_fk, x_start, y_start)
    rmse_re_fk = batch_moving_window_assignment(e_fkdnn, rmse_re_fk[np.newaxis, ...], x_start, y_start)[0, ...]
    

# Plot
#Resampling positions
plot_cscan(scanmap_rernd, 'Resample rnd', dx, dx, vmin = None, vmax = None)
plot_cscan(scanmap_rekrig, 'Resample krig', dx, dx, vmin = None, vmax = None)

# rmse_fkdnn
plot_cscan(rmse_re_fk, 'RMSE', dx, dx, vmin = None, vmax = None)

# Data
#plot_cscan(convert2cscan(A_re_fk), 'A_re_fk', dx, dx, vmin = None, vmax = None)
#plot_cscan(convert2cscan(A_re_idw), 'A_re_idw', dx, dx, vmin = None, vmax = None)

# Smoothed data
#plot_cscan(convert2cscan(gaussian_smoothing(A_re_fk, M, sigma = 5)), 'Smoothed A_re_fk', dx, dx)
#plot_cscan(convert2cscan(gaussian_smoothing(A_re_idw, M, sigma = 5)), 'Smoothed A_re_idw', dx, dx)


#%% SAFT

# FWM setting
specimen_parameters = {# Specimen parameters
    'dxdata': dx * ureg.meter,
    'dydata': dx * ureg.meter,
    'c0': c0 * ureg.meter / ureg.second,
    'fS': fS * ureg.hertz,
    'Nxdata': Nx,
    'Nydata': Ny,
    'Ntdata': M,
    'Nxreco': Nx,
    'Nyreco': Ny,
    'Nzreco': M,
    #'anglex': 0 * ureg.degree,
    #'angley': 0 * ureg.degree,
    'opening_angle' : opening_angle* ureg.degree,
    'zImage': -zmin * dz * ureg.meter,
    'xImage': 0 * 0.5 * 10**-3 * ureg.meter,
    'yImage': 0 * 0.5 * 10**-3 * ureg.meter,
    't0': zmin* dt * ureg.second
    }

apodization_parameters = {
        'max_angle' : opening_angle * ureg.degree
        }
pulse_parameters = {
        'pulseLength': 6,
        'fCarrier': fC* ureg.hertz,
        'B': np.sqrt(alpha)/fC, # 0.5
        'fS': fS* ureg.hertz
        }

propagation_model = PropagationModel3DSingleMedium()
propagation_model.set_parameters(specimen_parameters)
propagation_model.set_apodizationmodel('Bartlett', apodization_parameters)
propagation_model.set_pulsemodel('Gaussian', pulse_parameters)

# Initialize SAFTEngine
saft_engine = SAFTEngine(propagation_model, matrix_type = 'NLevelBlockToeplitz', enable_file_IO = True)

# True
#R_true = calc_reco(A_roi, saft_engine)
R_true = np.load('npy_data/simulations/R_true.npy')
R_true = R_true / np.abs(R_true).max()

# Sampled
R_smp = calc_reco(A_smp, saft_engine)
R_smp = R_smp / np.abs(R_smp).max()

# FK DNN
R_fkdnn = calc_reco(A_fkdnn, saft_engine)
R_fkdnn = R_fkdnn / np.abs(R_fkdnn).max()

# Hybrid
#R_hybrid = calc_reco(A_hybrid, saft_engine)
#R_hybrid = R_hybrid / np.abs(R_hybrid).max()

# IDW
R_idw = calc_reco(A_idw, saft_engine)
R_idw = R_idw / np.abs(R_idw).max()

### Resampled ones ###
# Resampled IDW
R_re_idw = calc_reco(A_re_idw, saft_engine)
R_re_idw = R_re_idw / np.abs(R_re_idw).max()

# Resampled FK
R_re_fk = calc_reco(A_re_fk, saft_engine)
R_re_fk = R_re_fk / np.abs(R_re_fk).max()


#%% Plots

import sys
sys.exit()

### C-Scan ###
plot_cscan(convert2cscan(R_true), 'Reco true', dx, dx, vmin = None, vmax = None)
plot_cscan(convert2cscan(R_smp), 'Reco smp', dx, dx, vmin = None, vmax = None)
#plot_cscan(convert2cscan(R_hybrid), 'Reco hybrid', dx, dx, vmin = None, vmax = None)
plot_cscan(convert2cscan(R_idw), 'Reco IDW', dx, dx, vmin = None, vmax = None)
plot_cscan(convert2cscan(R_fkdnn), 'Reco FKDNN', dx, dx, vmin = None, vmax = None)

# After resampling
plot_cscan(convert2cscan(R_re_idw), 'R_re_idw', dx, dx, vmin = None, vmax = None)
plot_cscan(convert2cscan(R_re_fk), 'R_re_fk', dx, dx, vmin = None, vmax = None)


### Side view###
plot_bscan(np.max(np.abs(R_true), axis = 2), 'Reco true', dz, dx, vmin = None, vmax = None)
plot_bscan(np.max(np.abs(R_smp), axis = 2), 'R_smp', dz, dx, vmin = None, vmax = None)
plot_bscan(np.max(np.abs(R_idw), axis = 2), 'R_idw', dz, dx, vmin = None, vmax = None)
plot_bscan(np.max(np.abs(R_fkdnn), axis = 2), 'R_fkdnn', dz, dx, vmin = None, vmax = None)

plot_bscan(np.max(np.abs(R_re_idw), axis = 2), 'R_re_idw', dz, dx, vmin = None, vmax = None)
plot_bscan(np.max(np.abs(R_re_fk), axis = 2), 'R_re_fk', dz, dx, vmin = None, vmax = None)



plot_bscan(R_true[:, :, 34], 'Reco true', dz, dx, vmin = None, vmax = None)
plot_bscan(R_smp[:, :, 34], 'R_smp', dz, dx, vmin = None, vmax = None)
plot_bscan(R_idw[:, :, 34], 'R_idw', dz, dx, vmin = None, vmax = None)
plot_bscan(R_fkdnn[:, :, 34], 'R_fkdnn', dz, dx, vmin = None, vmax = None)

plot_bscan(R_re_idw[:, :, 34], 'R_re_idw', dz, dx, vmin = None, vmax = None)
plot_bscan(R_re_fk[:, :, 34], 'R_re_fk', dz, dx, vmin = None, vmax = None)


## 3D plots
z_axis = np.arange(M)#* dz* 10**3 #[mm]
x_axis = np.arange(Nx)#* dx* 10**3 #[mm]
y_axis = np.flip(np.arange(Ny))#* dx* 10**3 #[mm]
z_label = 'z'
x_label = 'x'
y_label = 'y'

import sys
sys.exit()

SliceFigure3D(A_hybrid, 
              0, 'A_hybrid', 
              [z_axis, x_axis, y_axis], 
              [z_label, x_label, y_label], 
              60, False, display_text_info = True, info_height = 0.35)


