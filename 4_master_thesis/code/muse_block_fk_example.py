#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Block Frequency Kriging 
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import time
from scipy import ndimage

#import skgstat as skg

from ultrasonic_imaging_python.visualization.slice_figures import SliceFigure3D

from tools.display_time import display_time

from smart_inspect_data_formatter import SmartInspectDataFormatter
from spatial_subsampling import batch_subsampling
from spatial_subsampling import batch_random_walk2D
from spatial_subsampling import get_all_grid_points

plt.close('all') 

#%% Functions 

def plot_reco(data, title, dx, dz):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(data)
    ax.set_aspect(dz/dx)
    ax.set_title(title)
    ax.set(xlabel = 'x / dx', ylabel = 'z / dz')
    plt.colorbar(im)
    del fig
 
    
def plot_cscan(data, title, dx, dy, vmin = None, vmax = None):
    # !!!!!!! Swap axes to align to the MUSE CAD image !!!!!
    cscan = np.swapaxes(data, 0, 1)
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cscan, vmin = vmin, vmax = vmax)#, extent=[xrange[0], xrange[-1], yrange[0], yrange[-1]])
    ax.set_aspect(dy/dx)
    ax.set_title(title)
    ax.set(xlabel = 'x / dx', ylabel = 'y / dy')
    ax.invert_yaxis()
    plt.colorbar(im)
    
    del fig
    


#%% Parameter Setting 
# Variables: ROI
zmin, zmax = 1888, 2400 # = Nt_offset, Nt, 1865....1895 = only noise ...1865, 2020
xmin, xmax = 240, 350 #267, 297
ymin, ymax = 115, 165 #160
M = zmax - zmin
Nx = xmax - xmin
Ny = ymax - ymin

# Measurement setting
c0 = 5900 #[m/S]
fS = 80*10**6 #[Hz]  
fC = 3.36*10**6 #[Hz] 
D = 10*10**-3 #[m], transducer element diameter
opening_angle = 170#get_opening_angle(c0, D, fC)
dx = 0.5*10**-3 #[m], dy takes also the same value
dz = 3.6875*10**-5 #[m]


#%% MUSE data 
""" Info regarding the measurement data
* Raw data is flipped w.r.t. y-axis (compared to the CAD image)
    -> use np.flip(data, axis = 2)
* Back-wall echoes are removed
* Displaying C-Scan: use transpose, i.e. np.max(np.abs(data), axis = 0).T

"""
# Load
muse_data = scio.loadmat('MUSE/measurement_data.mat')
data = muse_data['data']
cscan = np.max(np.abs(data), axis = 0)

# plot_cscan(1, cscan, 'Data in C-Scan', dx, dx)

del muse_data

# Select the ROI
data_roi = data[zmin:zmax, xmin:xmax, ymin:ymax]
data_roi = data_roi / np.abs(data_roi).max() # normalize
del data

#plot_cscan(np.max(np.abs(data_roi), axis = 0).T, 'Data in our ROI (C-Scan)', dx, dx)

#%% Parameters
# Parameters for selecting the batch
N_batch = 10
x_start = 53#max(0, np.random.randint(Nx - N_batch)) #50, 75, 27, 48
y_start = 20#max(0, np.random.randint(Ny - N_batch)) #20, 31, 4, 38

# Parameters for spatial subsampling
coverage = 0.15#np.random.uniform(0.05, 0.101)#0.8
N_scan = int(coverage* N_batch**2)
seedNo = 493#np.random.randint(600) #277,  543, 270, 236, 71

# Position to predict its A-Scan
p0 = np.array([[5, 5]])#np.array([[int(N_batch/2), int(N_batch/2)]]) # Should be in an array form for scdis.cdist
s0 = np.around(dx* p0, 10)

# Parameters for FK & FV
fmin, fmax = 0, 200 
f_range = [fmin, fmax] # corresponds to bin (i.e. index), None if we don't want to limit the freq. range
maxlag = np.around(dx* N_batch/2* np.sqrt(2), 10) # in [m]
deg = 5 # Polynomial degree for FV
# Tikhonov regularization for Kriging weights calculation
alpha = 0.001*10**3

#%% Reduce data size into a batch
data_batch = data_roi[:, x_start:(x_start + N_batch), y_start: (y_start + N_batch)]
#data_batch = data[zmin:zmax, x_start:(x_start + N_batch), y_start: (y_start + N_batch)]

plot_cscan(np.max(np.abs(data_batch), axis = 0).T, 
            'Actual batch data (C-Scan) @ x_start = {}, y_start = {}'.format(x_start, y_start), 
            dx, dx)

#%% Spatial sub-sampling
# Position selection
p_smp = batch_subsampling(N_batch, N_scan, seed = seedNo) # Sampling positions, unitless
#p_smp = batch_random_walk2D(N_scan, N_batch, seed = seedNo)
# Remove s0 from the sampled positions
idx_del = np.argwhere(np.logical_and(p_smp[:, 0] == p0[0, 0], p_smp[:, 1] == p0[0, 1]))
# Check if s0 is in the sampled locations
if len(idx_del) != 0:
    idx_del = idx_del[0, 0]
    p_smp = np.delete(p_smp, idx_del, axis = 0) # Remove s0 from the sampled positions
# Convert into metre
s_smp = np.around(dx* p_smp, 10) # in [m]

# Scan map
scan_map = np.zeros((N_batch, N_batch))
scan_map[p_smp[:, 0], p_smp[:, 1]] = 1
scan_map[p0[0, 0], p0[0, 1]] = -1
plot_cscan(scan_map.T, 
           'Scan positions @ x_start = {}, y_start = {}, coverage = ca. {}'.format(x_start, y_start, coverage), 
           dx, dx)

# Measurement data of the selected sampling positions
formatter = SmartInspectDataFormatter(p_smp, 
                                      xmin = 0, xmax = N_batch, 
                                      ymin = 0, ymax = N_batch)
A_smp_2d = data_batch[:, p_smp[:, 0], p_smp[:, 1]] # shape = M y Ns
A_smp_3d = formatter.get_data_matrix_pixelwise(A_smp_2d) # size = M x N_batch x N_batch


#raise ValueError('Stop!')

#%% Block FK based on fv_full
from frequency_variogram import FrequencyVariogramRaw

# Full scan positions
p_full = get_all_grid_points(N_batch, N_batch) # unitless
s_full = np.around(dx* p_full, 10) # in [m]

# Compute FV full
cfvfull = FrequencyVariogramRaw(dx, maxlag)
cfvfull.set_positions(s_full)
cfvfull.set_data(data_batch, M, f_range = [fmin, fmax])
cfvfull.compute_fv()
cfvfull.smooth_fv(deg, ret_normalized = False)
fv_full = cfvfull.get_fv()
lags_full = cfvfull.get_lags() # Check if it is rounded to 10**-10!!! otherwise, error finding the correct indices!!!!

# Compute the freq. response of the samples
P_smp_2d = np.fft.rfft(A_smp_2d, M, axis = 0)[fmin:fmax, :] # shape = Nf x Ns

# Block FK
from frequency_kriging import FrequencyKrigingBlock
# Initialize the block FK class
bfk = FrequencyKrigingBlock(s_smp, N_batch, dx, lags_full) 
# Use true FV    fv, A_smp_2d, tik_factor, f_range = None):
Ahat_full = bfk.predict(fv_full, A_smp_2d, alpha, f_range = f_range) # shape = M x N_batch x N_bacth
E_full = bfk.get_prediction_error() # shape = Nf x Nc


plt.figure()
plt.plot(data_batch[:, p0[0, 0], p0[0, 1]], label = 'true')
plt.plot(Ahat_full[:, p0[0, 0], p0[0, 1]], label = 'block FK full') 
plt.legend()
plt.title('Predicted A-Scan @ ({}, {}) using the true FV of the RoI'.format(p0[0, 1], p0[0, 0]))
plt.xlabel('t / dt (excluding time offset)')
plt.ylabel('Amplitude')

plot_cscan(np.nanmax(Ahat_full, axis = 0).T, 'C_scan of predicted data: FK using true FV', dx, dx)


del bfk

raise ValueError('Stop!')

#%% Block FK based on FV DNN
from frequency_variogram import FrequencyVariogramDNN
#import tensorflow as tf
from tensorflow import keras

### FV estimation ###
# FVDNN class
cfvsmpDNN = FrequencyVariogramDNN(dx, N_batch, maxlag = maxlag)
cfvsmpDNN.set_positions(s_smp) 
cfvsmpDNN.feature_mapping(A_smp_2d, M, 3, deg, f_range = [fmin, fmax]) 
# FV prediction
modelname = 'tf_models/conv1d_{}/model'.format('L1reg_210512')
model = keras.models.load_model(modelname)
cfvsmpDNN.compute_fv(model)
fvnorm_smpDNN = cfvsmpDNN.get_fv()
fv_smpDNN = cfvsmpDNN.denormalize_fv(fvnorm_smpDNN)

def compare_fv(f_bin):
    plt.figure()
    plt.plot(lags_full, fv_full[f_bin, :], label = 'full')
    plt.plot(lags_full, fv_smpDNN[f_bin, :], label = 'smp DNN (L1)')
    plt.legend()
    plt.title('Comparison of FV for f_bin = {}'.format(f_bin))
    plt.xlabel('lag')
      

### Block FK ###
bfk = FrequencyKrigingBlock(s_smp, N_batch, dx, lags_full) 
Ahat_DNN = bfk.predict(fv_smpDNN, A_smp_2d, alpha, f_range = f_range) # shape = M x N_batch x N_bacth
rmse_FKDNN = bfk.get_prediction_error() # shape = N_batch x N_batch

# True RMSE
rmse_true = np.sum(np.abs(data_batch - Ahat_DNN), axis = 0) # shape = N_batch x N_batch

plt.figure()
plt.plot(data_batch[:, p0[0, 0], p0[0, 1]], label = 'true')
plt.plot(Ahat_DNN[:, p0[0, 0], p0[0, 1]], label = 'block FK DNN') 
plt.legend()

plot_cscan(rmse_FKDNN.T, 'Estimated RMSE of FK DNN', dx, dx)
plot_cscan(rmse_true.T, 'True RMSE of FK DNN', dx, dx)

plot_cscan(np.nanmax(Ahat_DNN, axis = 0).T, 'C_scan: Ahat_dnn', dx, dx)

#%% Block IDW
from general_batch_interpolation import InverseDistanceWeighting

bidw = InverseDistanceWeighting(s_smp, A_smp_2d, maxlag)
Ahat_idw = bidw.predict_blockwise(N_batch, dx)
rmse_idw = np.sum(np.abs(data_batch - Ahat_idw), axis = 0)

plt.figure()
plt.plot(data_batch[:, p0[0, 0], p0[0, 1]], label = 'true')
plt.plot(Ahat_idw[:, p0[0, 0], p0[0, 1]], label = 'block IDW') 
plt.legend()
plt.legend()
plt.title('Predicted A-Scan @ ({}, {}) via IDW'.format(p0[0, 1], p0[0, 0]))
plt.xlabel('t / dt (excluding time offset)')
plt.ylabel('Amplitude')


plot_cscan(rmse_idw.T, 'True RMSE of IDW', dx, dx)
plot_cscan(np.nanmax(Ahat_idw, axis = 0).T, 'C_scan: Ahat_idw', dx, dx)


#%% Plots
#import sys
#sys.exit()


z_axis = np.arange(M)#* dz* 10**3 #[mm]
x_axis = np.arange(N_batch)#* dx* 10**3 #[mm]
y_axis = np.flip(np.arange(N_batch))#* dx* 10**3 #[mm]
z_label = 'z'
x_label = 'x'
y_label = 'y'

### Along z-axis ###
SliceFigure3D(data_batch, 
              0, 'Ground truth', 
              [z_axis, x_axis, y_axis], 
              [z_label, x_label, y_label], 
              60, False, display_text_info = True, info_height = 0.35)
SliceFigure3D(Ahat_DNN, 
              0, 'FK DNN', 
              [z_axis, x_axis, y_axis], 
              [z_label, x_label, y_label], 
              60, False, display_text_info = True, info_height = 0.35)
SliceFigure3D(Ahat_idw, 
              0, 'IDW', 
              [z_axis, x_axis, y_axis], 
              [z_label, x_label, y_label], 
              60, False, display_text_info = True, info_height = 0.35)
    



#fv
f_bin = 7
plt.figure()
plt.plot(lags_full, fv_full[f_bin, :], label = 'fv_full')
plt.plot(lags_full, fv_dnn[f_bin, :], label = 'fv_dnn')
plt.plot(lags_full, fv_smp[f_bin, :], label = 'fv_smp')
plt.legend()
plt.xlabel('lags in [m]')
plt.ylabel('fv amplitude')
plt.title('true fv vs smoothed experimental fv for f_bin = {}'.format(f_bin))
#plt.title('true fv vs DNN estimated fv for f_bin = {}'.format(f_bin))


