# -*- coding: utf-8 -*-
"""
Kriging example using MUSE datasets
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
    
    
def plot_spatiotemporal_variogram(data, title, M, bins):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(data)
    ax.set_aspect(int(M/len(bins)))
    ax.set_title(title)
    ax.set(xlabel = 'z / dz', ylabel = 'distance [mm]')
    ax.set_yticklabels(np.around(bins, 3))
    plt.colorbar(im)
    del fig


def plot_variogram(data, title, bins):
    plt.figure()
    plt.plot(np.around(bins, 2), data)
    plt.title(title)
    plt.xlabel('distance [mm]')
    plt.ylabel('Semivariance')
    


def variogram_single_slice(coords, values, n_lags, estimator, ret_bins = False, ret_var_model = False):
    V = skg.Variogram(coords, values, n_lags = n_lags)
    #V.estimator = estimator
    var_exp = V.experimental # = experimental variogram
    bins = V.bins
    # Model the variogram = theoretical variogram
    x = np.linspace(0, np.nanmax(bins), 100)
    y = V.transform(x)
    var_model = np.array([x, y]).T
    
    V.plot(hist=False)
    
    del V
    
    if ret_bins == False and ret_var_model == False:
        return var_exp
    elif ret_bins == True and ret_var_model == False:
        print('return bins!')
        return bins, var_exp
    elif ret_bins == False and ret_var_model == True:
        return var_exp, var_model
    else:
        return bins, var_exp, var_model
        

    
#%% Parameter Setting 
# Variables: ROI
zmin, zmax = 1865, 2400 # = Nt_offset, Nt, 1865....1895 = only noise ...1865, 2020
xmin, xmax = 245, 350 #267, 297
ymin, ymax = 120, 165 #160
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
data_roi = data_roi / np.abs(data_roi).max() # normalize

plot_cscan(np.max(np.abs(data_roi), axis = 0).T, 'Data in our ROI (C-Scan)', dx, dx)

#%% Sampling
# Parameters
N_batch = 10
coverage = 1.0
N_scan = int(coverage* N_batch**2)

# Starting point for the batch
x_start = 50#max(0, np.random.randint(Nx - N_batch)) #21
y_start = 20#max(0, np.random.randint(Ny - N_batch)) #7


# Sampling positions
# All grid points within our ROI
xx, yy = np.meshgrid(np.arange(N_batch), np.arange(N_batch))
xx = xx.flatten()
yy = yy.flatten()
coords_full = np.array([xx.flatten(), yy.flatten()]).T
# Select the sampling coordinates
np.random.seed()#42
scan_idx = np.unique(np.random.randint(len(xx), size = N_scan + 10000))
scan_idx = scan_idx[:N_scan] # s.t. len(scan_idx) == N_scan
coords_x = xx[scan_idx]
coords_y = yy[scan_idx]
coords = np.array([coords_x, coords_y]).T

# Measurement data of the selected sampling positions
formatter = SmartInspectDataFormatter(coords, 
                                      xmin = 0, xmax = 0 + N_batch, 
                                      ymin = 0, ymax = 0 + N_batch)
data_sampled = formatter.get_data_matrix_pixelwise(data_roi[:, x_start + coords_x, y_start + coords_y]) # size = M x N_batch x N_batch



#%% Experimental variogram for a SINGLE slice
# Variogram parameters
n_lags = 10 # makes difference, when the number of samples is reduced (?)
model = 'gaussian'
maxlag = None#0.5*np.sqrt(2)* N_batch

# Full batch data
np.random.seed()
sliceNo_z = 66#np.random.randint(M) #465 -> seems to fit very nicely with the gaussian model
data_batch = data_roi[sliceNo_z, x_start:(x_start + N_batch), y_start: (y_start + N_batch)]
#data_batch = data_batch / np.abs(data_batch).max() # normalize


# Full data
values = data_batch[xx, yy] 
V_full = skg.Variogram(coords_full, values, n_lags = n_lags, model = model, maxlag = maxlag)
bins = V_full.bins
x = np.linspace(0, np.nanmax(bins), 100)
y_full = V_full.transform(x)
V_full.plot(hist = False)


# Sampled data
values = data_batch[coords_x, coords_y]
V_sampled = skg.Variogram(coords, values, n_lags = n_lags, model = model, maxlag = maxlag)
y_sampled = V_sampled.transform(x)
V_sampled.plot(hist = False)


# Plot variogram models
plt.figure()
plt.plot(x, y_full, label = 'full data')
plt.plot(x, y_sampled, label = 'sampled data')
plt.legend()
plt.xlabel('normalized distance')
plt.ylabel('Semivariance (model)')


#%% Kriging
ok = skg.OrdinaryKriging(V_sampled, min_points = 1)
field = ok.transform(coords_full[:, 0], coords_full[:, 1])
field = np.reshape(field, (N_batch, N_batch), 'F')
sigma = np.reshape(ok.sigma, (N_batch, N_batch), 'F')

vmin, vmax = data_batch.min(), data_batch.max()

plot_cscan(field.T, 'Interpolated data', dx, dx, vmin = vmin, vmax = vmax)
#plot_cscan(sigma.T, 'Estimated kriging error', dx, dx, vmin = vmin, vmax = vmax)
plot_cscan((data_batch - field).T, 'Actual kriging error', dx, dx, vmin = vmin, vmax = vmax)
plot_cscan(data_batch.T, 'Actual data', dx, dx, vmin = vmin, vmax = vmax)
plot_cscan(data_sampled[sliceNo_z, :, :].T, 
           'Sampled data @ x_start = {}, y_start = {}, coverage = ca. {}'.format(x_start, y_start, coverage), 
           dx, dx, vmin = vmin, vmax = vmax)

#%% Plots: data
z_axis = np.arange(M)#* dz* 10**3 #[mm]
x_axis = np.arange(Nx)#* dx* 10**3 #[mm]
y_axis = np.flip(np.arange(Ny))#* dx* 10**3 #[mm]
z_label = 'z'
x_label = 'x'
y_label = 'y'

# Sampled data in the selected batch
fig_z = SliceFigure3D(data_sampled, 
                      0, 'Sampled data slice (@ x_start = {}, y_start = {}) along z-axis'.format(x_start, y_start), 
                      [z_axis, np.arange(N_batch), np.flip(np.arange(N_batch))], 
                      [z_label, x_label, y_label], 
                      sliceNo_z, False, display_text_info = True, info_height = 0.35)


# Batch slice along z-axis
fig_z = SliceFigure3D(data_roi[:, x_start:(x_start + N_batch), y_start: (y_start + N_batch)], 
                      0, 'Batch slice (@ x_start = {}, y_start = {}) along z-axis'.format(x_start, y_start), 
                      [z_axis, np.arange(N_batch), np.flip(np.arange(N_batch))], 
                      [z_label, x_label, y_label], 
                      sliceNo_z, False, display_text_info = True, info_height = 0.35)

# Slice along z-axis
fig_z = SliceFigure3D(data_roi, 0, 'MUSE data slice along z-axis', 
                      [z_axis, x_axis, y_axis], [z_label, x_label, y_label], 
                      sliceNo_z, False, display_text_info = True, info_height = 0.35)


plt.show()


