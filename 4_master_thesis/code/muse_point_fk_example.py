#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Block Frequency Kriging 
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import scipy.signal as scsi
import numpy.fft as fft
import scipy.spatial.distance as scdis
import time
from numpy.polynomial import Polynomial

#import skgstat as skg

from ultrasonic_imaging_python.visualization.slice_figures import SliceFigure3D

from tools.display_time import display_time

from smart_inspect_data_formatter import SmartInspectDataFormatter
from spatial_subsampling import get_all_grid_points
from spatial_subsampling import batch_subsampling


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
 
    
def plot_cscan(data, title, dx, dy):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(data)
    ax.set_aspect(dy/dx)
    ax.set_title(title)
    ax.set(xlabel = 'x / dx', ylabel = 'y / dy')
    plt.colorbar(im)
    del fig
    
    
#%% MUSE data and its parameters
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

# MUSE data
""" Info regarding the measurement data
* Raw data is flipped w.r.t. y-axis (compared to the CAD image)
    -> use np.flip(data, axis = 2)
* Back-wall echoes are removed
* Displaying C-Scan: use transpose, i.e. np.max(np.abs(data), axis = 0).T

"""
# Load
muse_data = scio.loadmat('MUSE/measurement_data.mat')
data = np.flip(muse_data['data'], axis = 2) # Raw data: order is alligned to the CAD image 
cscan = np.max(np.abs(data), axis = 0).T # To align the CAD image

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
x_start = 41#max(0, np.random.randint(Nx - N_batch)) #50, 75, 27, 48
y_start = 20#max(0, np.random.randint(Ny - N_batch)) #20, 31, 4, 38

# Parameters for spatial subsampling
coverage = 0.3#np.random.uniform(0.05, 0.101)#0.8
N_scan = 15#int(coverage* N_batch**2)
seedNo = 71#np.random.randint(600) #277,  543, 270, 236, 71

# Position to predict its A-Scan
p0 = np.array([[2, 7]])#np.array([[int(N_batch/2), int(N_batch/2)]]) # Should be in an array form for scdis.cdist
s0 = np.around(dx* p0, 10) # in [m]

# Parameters for FK & FV
[fmin, fmax] = [0, 50] #[fmin, fmax], corresponds to bin (i.e. index), None if we don't want to limit the freq. range
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

#%% Compute FV
from frequency_variogram import FrequencyVariogramRaw, FrequencyVariogramDNN

#### Using full grid points ###
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


### Using sampled points: FV estimation via DNN ###
from tensorflow import keras

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
    
    
compare_fv(23)
compare_fv(24)
compare_fv(25)
compare_fv(48)


#%% Point FK
from frequency_kriging import FrequencyKrigingPoint

# Compute the freq. response of the samples
P_smp_2d = np.fft.rfft(A_smp_2d, M, axis = 0)[fmin:fmax, :] # shape = Nf x Ns

# Initialize the block FK class
pfk = FrequencyKrigingPoint(s_smp, N_batch, dx, lags_full) 
pfk.set_prediction_position(s0)

### FV full ###
p0hat_full = pfk.predict(fv_full, P_smp_2d, alpha)
a0hat_full = np.fft.irfft(p0hat_full, n = M, axis = 0)
e_full = pfk.prediction_error()

plt.figure()
plt.plot(data_batch[:, p0[0, 0], p0[0, 1]], label = 'true')
plt.plot(a0hat_full, label = 'Point FK full') 
plt.legend()

### FV DNN ###
p0hat_DNN = pfk.predict(fv_smpDNN, P_smp_2d, alpha)
a0hat_DNN = np.fft.irfft(p0hat_DNN, n = M, axis = 0)
e_full = pfk.prediction_error()

plt.figure()
plt.plot(data_batch[:, p0[0, 0], p0[0, 1]], label = 'true')
plt.plot(a0hat_DNN, label = 'Point FK DNN') 
plt.legend()


#%% Inverse-Distance Weighting (IDW)
# Limit the range with the maxlag
idx_Svalid = pfk.idx_Svalid
S_valid = pfk.S_valid
A_valid_2d = A_smp_2d[:, idx_Svalid]

# Calculate the lags
lags0 = np.linalg.norm((S_valid - s0), axis = 1)

w_idw = 1/lags0
w_idw = w_idw/np.sum(w_idw)

a0hat_idw = np.dot(A_valid_2d, w_idw)

# Show the l2 norm of the error
err_idw = data_batch[:, p0[0, 0], p0[0, 1]] - a0hat_idw
print('IDW: prediction error (l2) = {}'.format(np.round(np.linalg.norm(err_idw)**2, 5)))

#%% Plots

plt.figure()
plt.plot(data_batch[:, p0[0, 0], p0[0, 1]], label = 'true')
plt.plot(a0hat_idw, label = 'IDW')
plt.legend()
plt.title('Ascan prediction via IDW')
plt.xlabel('t/dt')
plt.ylabel('Amplitude')


plt.figure()
plt.plot(data_batch[:, p0[0, 0], p0[0, 1]] - a0hat_idw, label = 'err_idw')
plt.plot(data_batch[:, p0[0, 0], p0[0, 1]] - a0hat_DNN, label = 'err_FK_DNN')
plt.legend()
plt.title('Prediction error')
plt.xlabel('t/dt')
plt.ylabel('Amplitude')

