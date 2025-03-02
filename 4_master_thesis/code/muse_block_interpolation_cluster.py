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
 
    
def plot_cscan(data_3d, title, dx, dy, vmin = None, vmax = None):
    # Compute the C-san
    cscan = np.max(np.abs(data_3d), axis = 0)
    
    # !!!!!!! Swap axes to align to the MUSE CAD image !!!!!
    cscan = np.swapaxes(cscan, 0, 1)
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cscan, vmin = vmin, vmax = vmax)#, extent=[xrange[0], xrange[-1], yrange[0], yrange[-1]])
    ax.set_aspect(dy/dx)
    ax.set_title(title)
    ax.set(xlabel = 'x / dx', ylabel = 'y / dy')
    ax.invert_yaxis()
    plt.colorbar(im)
    
    del fig
    

    
#%% Load the MUSE data in the selected batch
# Batch ROI in MUSE specimen: [xmin : xmax] = 293, 303 & [ymin : ymax] = 135, 145
# True A-Scans, scan positions, Sampled A-Scans
A_true = np.load('npy_data/batch_itp/A_true.npy') # M x N_batch x N_batch
p_smp = np.load('npy_data/batch_itp/p_smp.npy') # N_scan = number of scans
A_smp = np.load('npy_data/batch_itp/A_smp.npy') # M x N_batch x N_batch

# Dimensions
(M, N_batch, _) = A_true.shape
N_scan = p_smp.shape[0]

#plot_cscan(A_true, 'Actual batch data (C-Scan)', 1, 1)


#%% Parameter Setting 
# Variables: ROI
zmin, zmax = 1888, 2400 # = Nt_offset, Nt, 1865....1895 = only noise ...1865, 2020
# Check if M == zmax - zmin
if M != (zmax - zmin):
    raise ValueError('Temporal dimensions do not match!')

# Measurement setting
c0 = 5900 #[m/S]
fS = 80*10**6 #[Hz]  
fC = 3.36*10**6 #[Hz] 
D = 10*10**-3 #[m], transducer element diameter
opening_angle = 170#get_opening_angle(c0, D, fC)
dx = 0.5*10**-3 #[m], dy takes also the same value
dz = 3.6875*10**-5 #[m]

# Parameters for FK & FV
fmin, fmax = 0, 200 
f_range = [fmin, fmax] # corresponds to bin (i.e. index), None if we don't want to limit the freq. range
maxlag = np.around(dx* N_batch/2* np.sqrt(2), 10) # in [m]
deg = 5 # Polynomial degree for FV
# Tikhonov regularization for Kriging weights calculation
alpha = 0.001*10**3

#%% Save the data as the matlab formats
# Data to save = A_ture, sampling positions in mesh grids, flattened meas. data, query points, dx, dz, N_batch, M
# !! Query points are computed in matlab using meshgrid
# Sampling positions in mesh grid form
# Indices (i.e. no unit!!!)
z_smp = np.tile(np.arange(M), N_scan) # format = [0...M-1, 0...M-2, ...], shape = M* N_scan
x_smp = np.repeat(p_smp[:, 0], M) # format = [x0, x0, ...., x1, x1.....,], shape = M* N_scan
y_smp = np.repeat(p_smp[:, 1], M) # format = [x0, x0, ...., x1, x1.....,], shape = M* N_scan

# Measurement data (i.e. sampled values)
V_smp = A_true[:, p_smp[:, 0], p_smp[:, 1]].flatten('F') # size = M* N_batch**2

# Query points
# Unfolding "directions"
# The mesh grids are unfolded in the following manner
# (1) Take a slice along y-axis
# (2) Vectorize the selected slice s.t. the z-values increase faster than x-values
# (3) Go to the next slice
(x_quer, y_quer, z_quer) = np.meshgrid(np.arange(N_batch), np.arange(N_batch), np.arange(M)) 
x_quer = x_quer.flatten('C')
y_quer = y_quer.flatten('C')
z_quer = z_quer.flatten('C')

# size = N_scan x N_scan x M


import sys
sys.exit()

# Save data as .mat format -> need to be a dictionary! 
mdict = {
    'dx' : dx,
    'dz' : dz,
    'N_batch' : N_batch,
    'M' : M,
    'A_true' : A_true,
    'V_smp' : V_smp,
    'x_smp' : x_smp,
    'y_smp' : y_smp,
    'z_smp' : z_smp,
    'x_quer' : x_quer.flatten('C'),
    'y_quer' : y_quer.flatten('C'),
    'z_quer' : z_quer.flatten('C')
    }

from scipy.io import savemat
fname = 'npy_data/batch_itp.mat'
savemat(fname, mdict)
print('!! mat data saved as {} !!'.format(fname))




