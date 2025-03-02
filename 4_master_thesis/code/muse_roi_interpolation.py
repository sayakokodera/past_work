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
    ax.set_aspect(dz/dx)
    ax.set_title(title)
    ax.set(xlabel = 'x / dx', ylabel = 'z / dz')
    plt.colorbar(im)
    
    del fig
    

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

    

def make_scanmap(p_smp_roi, Nx, Ny):
    scan_map = np.zeros((Nx, Ny))
    scan_map[p_smp_roi[:, 0], p_smp_roi[:, 1]] = 1
    return scan_map
    

def gaussian_smoothing(arr, M, sigma = 5):
    arrsmt = np.zeros(arr.shape)
    for row in range(M):
        arrsmt[row, :, :] = ndimage.gaussian_filter(arr[row, :, :], sigma = 5)
    
    return arrsmt
    


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
# Load re/sampling positions
coverage = 15
setNo = 21
p_smp = np.load('npy_data/simulations/210625_{}%/setNo_{}/p_smp.npy'.format(coverage, setNo))
p_resmp_krig = np.load('npy_data/simulations/210625_{}%/setNo_{}/p_resmp_krig.npy'.format(coverage, setNo))
p_resmp_rnd = np.load('npy_data/simulations/210625_{}%/setNo_{}/p_resmp_rnd.npy'.format(coverage, setNo))
rmse_fk = np.load('npy_data/simulations/210625_{}%/setNo_{}/rmse_FK.npy'.format(coverage, setNo))
rmse_re_fk = np.load('npy_data/simulations/210625_{}%/setNo_{}/rmse_re_FK.npy'.format(coverage, setNo))


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
[fmin, fmax] = [10, 40] #[fmin, fmax], corresponds to bin (i.e. index), None if we don't want to limit the freq. range
deg = 5 # Polynomial degree for FV
tik_alpha = 0.001*10**3 # Tikhonov regularization for Kriging weights calculation+
modelname = 'tf_models/conv1d_{}/model'.format('L1reg_210512') # Model for FV DNN
l_cnnwin = 3 # Window length fo CNN 1D


def batch_interpolation(p_start, scan_map, perform_FK = True):
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
        rmse_FKDNN[p_smp[0, 0], p_smp[0, 1]] = 0.0
          
    else:
        # Sampled data in the current batch
        A_smp_2d = A_roi[:, x_start + p_smp[:, 0], y_start + p_smp[:, 1]] # shape = M x Ns
        
        ### Batch IDW ###
        bidw = InverseDistanceWeighting(s_smp, A_smp_2d, maxlag)
        Ahat_idw = bidw.predict_blockwise(N_batch, dx)
        
        # Skip FK
        if perform_FK == False:
            Ahat_DNN = np.zeros((M, N_batch, N_batch))
            rmse_FKDNN = np.nan* np.ones((N_batch, N_batch))
        
        ### Batch FK ###
        else:    
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
            
            display_time(round(time.time() - start, 5))
     
    return Ahat_idw, Ahat_DNN, rmse_FKDNN
        
            
            
    
A_idw = np.zeros(A_roi.shape)
A_re_idw = np.zeros(A_roi.shape)
A_fk = np.zeros(A_roi.shape)
A_re_fk = np.zeros(A_roi.shape)

start_all = time.time()
for batchNo, p in enumerate(p_batch_all):
    # Current batch
    x_start, y_start = p
    #=========== With initial sampling points ======#
    print('With initial scan map!')
    # Interpolate
    arr_idw, arr_fkdnn, _ = batch_interpolation(p, make_scanmap(p_smp, Nx, Ny))
    # Assign: scaling according to moving window averaging -> assignment
    A_idw = batch_moving_window_assignment(arr_idw, A_idw, x_start, y_start)
    A_fk = batch_moving_window_assignment(arr_fkdnn, A_fk, x_start, y_start)
    
    #=========== With resampling points for FK ======#
    print('With resampling for FK!')
    # Interpolate
    _, arr_refk, _ = batch_interpolation(p, make_scanmap(p_resmp_krig, Nx, Ny))
    # Assign: scaling according to moving window averaging -> assignment
    A_re_fk = batch_moving_window_assignment(arr_refk, A_re_fk, x_start, y_start)
    
    #=========== With resampling points for IDW ======#
    print('With resampling for IDW!')
    # Interpolate
    arr_reidw, _, _ = batch_interpolation(p, make_scanmap(p_resmp_rnd, Nx, Ny), perform_FK = False)
    # Assign: scaling according to moving window averaging -> assignment
    A_re_idw = batch_moving_window_assignment(arr_reidw, A_re_idw, x_start, y_start)
    
       
display_time(round(time.time() - start_all, 5))


#%% Smoothing
Asmt_fk = gaussian_smoothing(A_fk, M, sigma = 5)
Asmt_re_fk = gaussian_smoothing(A_re_fk, M, sigma = 5)


#%% Plots

# 2D plots
plot_cscan(convert2cscan(A_roi), 'A_roi', dx, dx, vmin = None, vmax = None)
plot_cscan(convert2cscan(A_idw), 'A_idw', dx, dx, vmin = None, vmax = None)
plot_cscan(convert2cscan(A_re_idw), 'A_re_idw', dx, dx, vmin = None, vmax = None)
plot_cscan(convert2cscan(A_fk), 'A_fk', dx, dx, vmin = None, vmax = None)
plot_cscan(convert2cscan(A_re_fk), 'A_re_fk', dx, dx, vmin = None, vmax = None)
plot_cscan(convert2cscan(Asmt_fk), 'Asmt_fk', dx, dx, vmin = None, vmax = None)
plot_cscan(convert2cscan(Asmt_re_fk), 'Asmt_re_fk', dx, dx, vmin = None, vmax = None)

# B-Scan
sliceNo = 30
plot_bscan(A_roi[:, :, sliceNo], 'A_roi', dz, dx, vmin = None, vmax = None)
plot_bscan(A_idw[:, :, sliceNo], 'A_idw', dz, dx, vmin = None, vmax = None)
plot_bscan(A_re_idw[:, :, sliceNo], 'A_re_idw', dz, dx, vmin = None, vmax = None)
plot_bscan(A_fk[:, :, sliceNo], 'A_fk', dz, dx, vmin = None, vmax = None)
plot_bscan(A_re_fk[:, :, sliceNo], 'A_re_fk', dz, dx, vmin = None, vmax = None)
plot_bscan(Asmt_fk[:, :, sliceNo], 'Asmt_fk', dz, dx, vmin = None, vmax = None)
plot_bscan(Asmt_re_fk[:, :, sliceNo], 'Asmt_re_fk', dz, dx, vmin = None, vmax = None)

### 3D plots
#z_axis = np.arange(M)#* dz* 10**3 #[mm]
#x_axis = np.arange(Nx)#* dx* 10**3 #[mm]
#y_axis = np.flip(np.arange(Ny))#* dx* 10**3 #[mm]
#z_label = 'z'
#x_label = 'x'
#y_label = 'y'
#
#import sys
#sys.exit()
#
#SliceFigure3D(A_hybrid, 
#              0, 'A_hybrid', 
#              [z_axis, x_axis, y_axis], 
#              [z_label, x_label, y_label], 
#              60, False, display_text_info = True, info_height = 0.35)


