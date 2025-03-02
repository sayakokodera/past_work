# -*- coding: utf-8 -*-
"""
Spatio-temporal Kriging example using MUSE datasets
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import scipy.signal as scsi
import time

import skgstat as skg

from ultrasonic_imaging_python.visualization.slice_figures import SliceFigure3D

from tools.display_time import display_time
from smart_inspect_data_formatter import SmartInspectDataFormatter
from signal_denoising import denoise_svd

plt.close('all') 

#======================================================================================================= Functions ====#
    
def plot_cscan(data, title, dx, dy, vmin = None, vmax = None):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(data, vmin = vmin, vmax = vmax)
    ax.set_aspect(dy/dx)
    ax.set_title(title)
    ax.set(xlabel = 'x / dx', ylabel = 'y / dy')
    plt.colorbar(im)
    del fig
    
    
def plot_spatiotemporal_variogram(variogram, title, M, bins):
    """
    Parameters
    ----------
        variogram: array, row = distance bins, columns = temporal slice
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(variogram)
    ax.set_aspect(1/int(M/len(bins)))
    ax.set_title(title)
    ax.set(xlabel = 'distance', ylabel = 'z / dz')
    ax.set_xticks(np.arange(0, len(bins), 2))
    ax.set_xticklabels(np.around(bins[0::2], 3))
    plt.colorbar(im)
    del fig


def plot_variogram(data, title, bins):
    plt.figure()
    plt.plot(np.around(bins, 2), data)
    plt.title(title)
    plt.xlabel('distance [mm]')
    plt.ylabel('Semivariance')
    


def spatiotemporal_kriging(data, coords, n_lags, model = 'gaussian', maxlag = None, ret_Vdict = False):
    # Sampled positions
    coords_x, coords_y = coords[:, 0], coords[:, 1]
    # Temporal dimension
    M = data.shape[0]
    
    # All grid points within our ROI for interpolation
    xx, yy = np.meshgrid(np.arange(N_batch), np.arange(N_batch))
    xx = xx.flatten()
    yy = yy.flatten()
    coords_full = np.array([xx.flatten(), yy.flatten()]).T
    
    # Base
    Vari = np.zeros((M, n_lags))
    bins = np.zeros(n_lags)
    data_itp = np.copy(data)
    
    for sliceNo_z in range(M):
        print('Slice No.{}'.format(sliceNo_z))
        values = data[sliceNo_z, coords_x, coords_y]
        
        # Check if all values are same -> skip interpolation (otherwise skg.Variogram fails)
        if len(set(values)) < 2:
            pass
        
        else:
            # Experimental variogram
            start = time.time()
            V = skg.Variogram(coords, values, n_lags = n_lags,  model = model, maxlag = maxlag)
            vari = V.experimental
            #print('Variogram computation')
            #display_time(round(time.time() - start, 7))
            
            # Save variogram
            if ret_Vdict == True:
                Vari[sliceNo_z, :] = np.copy(vari)
                if sliceNo_z == 0:
                    bins = V.bins
            
            # Ignore noisy part (where variogram is more or less constant)
            if np.abs(vari).max() < 1.0:
                pass
            
            # Kriging
            else:
                start = time.time()
                ok = skg.OrdinaryKriging(V, min_points = 1)
                batch_hat = ok.transform(coords_full[:, 0], coords_full[:, 1])
                data_itp[sliceNo_z, :, :] = np.reshape(batch_hat, (N_batch, N_batch), 'F')
                print('Kriging computation')
                display_time(round(time.time() - start, 3))
    
    if ret_Vdict == False:
        return data_itp
    else:
        return data_itp, {'bins': bins, 'Variogram' : Vari}


    
#%% Parameter Setting 
# Variables: ROI
zmin, zmax = 1865, 2400 # = Nt_offset, Nt, 1865....1895 = only noise ...1865, 2020
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

# Model parameter (estimated from the measurement data)
alpha = 10.6*10**12 #[Hz]**2, bandwidth factor
r = 0.75 # Chirp rate
with_envelope = False

#%% MUSE data 
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
#data_roi = data_roi / np.abs(data_roi).max() # normalize

#plot_cscan(np.max(np.abs(data_roi), axis = 0).T, 'Data in our ROI (C-Scan)', dx, dx)

#%% Sampling
# Parameters
N_batch = 10 # Batch size (i.e. # of pixels along each axis within a single batch)
coverage = 0.3
N_scan = int(coverage* N_batch**2)

# Starting point for the batch
x_start = 50#max(0, np.random.randint(Nx - N_batch)) #21
y_start = 20#max(0, np.random.randint(Ny - N_batch)) #7

# Ground truth: fully sampled data
data_batch = data_roi[:, x_start:(x_start + N_batch), y_start: (y_start + N_batch)]


# Sampling positions
# All grid points within our ROI
xx, yy = np.meshgrid(np.arange(N_batch), np.arange(N_batch))
xx = xx.flatten()
yy = yy.flatten()
coords_full = np.array([xx.flatten(), yy.flatten()]).T
# Select the sampling coordinates
np.random.seed(729)#42, 52
scan_idx = np.unique(np.random.randint(len(xx), size = N_scan + 2000))
scan_idx = np.random.permutation(scan_idx)[:N_scan]
coords_x = xx[scan_idx]
coords_y = yy[scan_idx]
# Add scan positions to improve the accuracy
#coords_x = np.append(coords_x, [4, 6, 2])
#coords_y = np.append(coords_y, [5, 6, 6])
# Resuting scan positions
coords = np.array([coords_x, coords_y]).T
#coords = np.load('coords_x0y0.npy')

# Measurement data of the selected sampling positions
formatter = SmartInspectDataFormatter(coords, 
                                      xmin = 0, xmax = 0 + N_batch, 
                                      ymin = 0, ymax = 0 + N_batch)
data_sampled = formatter.get_data_matrix_pixelwise(data_batch[:, coords_x, coords_y]) # size = M x N_batch x N_batch


#%% Spatio-temporal kriging
# Variogram parameters
n_lags = 10 # makes difference, when the number of samples is reduced (?)
model = 'gaussian'
maxlag = 0.5* np.sqrt(2)* N_batch # maxlag to limit the "infulence" of the far away points => less weights to calc.

start= time.time()
# Spatiotemporal interpolation: temporal slice wise kriging
data_itp, V_dict = spatiotemporal_kriging(data_sampled, coords, n_lags, model = 'gaussian', 
                                          maxlag = maxlag, ret_Vdict = True)
err = data_batch - data_itp # Estimation error

display_time(round(time.time() - start, 3))
print('Actual coverage = {}%'.format(round(100* len(scan_idx)/(N_batch**2), 2)))

plot_spatiotemporal_variogram(V_dict['Variogram'][:, :-1], 'Variogram', M, V_dict['bins'][:-1])
print(V_dict['bins'])


#%% Plots: data
z_axis = np.flip(np.arange(M))#* dz* 10**3 #[mm]
x_axis = np.arange(N_batch)#* dx* 10**3 #[mm]
y_axis = np.arange(N_batch)#* dx* 10**3 #[mm]
z_label = 'z'
x_label = 'x'
y_label = 'y'

# Value range
data_min = min(data_batch.min(), np.nanmin(data_itp))
data_max = max(data_batch.max(), np.nanmax(data_itp))
print('Actual data_min = {}, data_max = {}'.format(data_batch.min(), data_batch.max()))
print('data min = {}, data_max = {}'.format(data_min, data_max))
 

# Sampled data in the selected batch
fig_z = SliceFigure3D(data_sampled, 
                      0, 'Sampled data slice (@ x_start = {}, y_start = {}) along z-axis'.format(x_start, y_start), 
                      [z_axis, x_axis, np.flip(y_axis)], 
                      [z_label, x_label, y_label], 
                      322, False, display_text_info = True, info_height = 0.35,
                      data_min = data_min, data_max = data_max)


# Interpolated batch slice along z-axis
fig_z = SliceFigure3D(data_itp, 
                      0, 'Interpolated slice (@ x_start = {}, y_start = {}) along z-axis'.format(x_start, y_start), 
                      [z_axis, x_axis, np.flip(y_axis)],
                      [z_label, x_label, y_label], 
                      322, False, display_text_info = True, info_height = 0.35,
                      data_min = data_min, data_max = data_max)


# Actual batch slice along z-axis
fig_z = SliceFigure3D(data_batch, 
                      0, 'Actual data slice (@ x_start = {}, y_start = {}) along z-axis'.format(x_start, y_start), 
                      [z_axis, x_axis, np.flip(y_axis)],
                      [z_label, x_label, y_label], 
                      322, False, display_text_info = True, info_height = 0.35,
                      data_min = data_min, data_max = data_max)

# Error slice along z-axis
fig_z = SliceFigure3D(err, 
                      0, 'Error slice (@ x_start = {}, y_start = {}) along z-axis'.format(x_start, y_start), 
                      [z_axis, x_axis, np.flip(y_axis)],
                      [z_label, x_label, y_label], 
                      322, False, display_text_info = True, info_height = 0.35,
                      data_min = data_min, data_max = data_max)

# =============================================================================
# # Slice along z-axis
# fig_z = SliceFigure3D(data_roi, 0, 'MUSE data slice along z-axis', 
#                       [z_axis, x_axis, y_axis], [z_label, x_label, y_label], 
#                       0, False, display_text_info = True, info_height = 0.35)
# =============================================================================


# Slice along y
# Sampled
fig_y = SliceFigure3D(data_sampled, 
                      2, 'Sampled data slice (@ x_start = {}, y_start = {}) along y-axis'.format(x_start, y_start), 
                      [z_axis, x_axis, y_axis], [z_label, x_label, y_label], 
                      0, False, display_text_info = True, info_height = 0.35,
                      data_min = data_min, data_max = data_max)

# Interpolated
fig_y = SliceFigure3D(data_itp, 
                      2, 'Interpolated data slice (@ x_start = {}, y_start = {}) along y-axis'.format(x_start, y_start), 
                      [z_axis, x_axis, y_axis], [z_label, x_label, y_label], 
                      0, False, display_text_info = True, info_height = 0.35,
                      data_min = data_min, data_max = data_max)

# Actual
fig_y = SliceFigure3D(data_batch, 
                      2, 'Actual data slice (@ x_start = {}, y_start = {}) along y-axis'.format(x_start, y_start), 
                      [z_axis, x_axis, y_axis], [z_label, x_label, y_label], 
                      0, False, display_text_info = True, info_height = 0.35,
                      data_min = data_min, data_max = data_max)

# Error 
fig_y = SliceFigure3D(err, 
                      2, 'Error slice (@ x_start = {}, y_start = {}) along y-axis'.format(x_start, y_start), 
                      [z_axis, x_axis, y_axis], [z_label, x_label, y_label], 
                      0, False, display_text_info = True, info_height = 0.35,
                      data_min = data_min, data_max = data_max)




# A-Scans
x, y = 3,8

# Interpolated vs actual
plt.figure()
plt.title('A-Scan @ (x, y) = ({}, {})'.format(x, y))
plt.plot(data_roi[:, (x_start + x), (y_start + y)], label = 'Actual')
plt.plot(data_itp[:, x, y], label = 'Interpolated')
plt.legend()

plt.show()


