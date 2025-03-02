# -*- coding: utf-8 -*-
"""
Spatio-temporal Kriging for a selected ROI in MUSE datasets
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import scipy.signal as scsi
import time
import json

import skgstat as skg

from ultrasonic_imaging_python.visualization.slice_figures import SliceFigure3D

from tools.display_time import display_time
from tools.npy_file_writer import save_data
from tools.datetime_formatter import DateTimeFormatter

from smart_inspect_data_formatter import SmartInspectDataFormatter
from spatial_subsampling import batch_subsampling

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
    warning = []
    
    for sliceNo_z in range(M):
        values = data[sliceNo_z, coords_x, coords_y]
        
        # Check if all values are same -> skip interpolation (otherwise skg.Variogram fails)
        if len(set(values)) < 2:
            pass
        
        else:
            # Experimental variogram
            #start = time.time()
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
                #start = time.time()
                ok = skg.OrdinaryKriging(V, min_points = 1)
                batch_hat = ok.transform(coords_full[:, 0], coords_full[:, 1])
                data_itp[sliceNo_z, :, :] = np.reshape(batch_hat, (N_batch, N_batch), 'F')
                #print('Kriging computation')
                #display_time(round(time.time() - start, 3))
                if ok.no_points_error > 0:
                    warning.append(sliceNo_z)
    
    if ret_Vdict == False:
        return data_itp, warning
    else:
        return data_itp, warning, {'bins': bins, 'Variogram' : Vari}


    
#%% Parameter Setting 
# Variables: ROI
zmin, zmax = 1865, 2400 # = Nt_offset, Nt, 1865....1895 = only noise ...1865, 2020
xmin, xmax = 240, 350 
ymin, ymax = 115, 165 
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

#%% Batch wise spatptemporal kriging
# Parameters w.r.t. batch
N_batch = 10 # Batch size (i.e. # of pixels along each axis within a single batch)
coverage = 0.3
N_scan = int(coverage* N_batch**2)


# Variogram parameters
n_lags = 10 # makes difference, when the number of samples is reduced (?)
model = 'gaussian'
maxlag = 0.5* np.sqrt(2)* N_batch # maxlag to limit the "infulence" of the far away points => less weights to calc.

# Params for file savnig
dtf = DateTimeFormatter()
f_date = dtf.get_date_str()

# Starting points for each batch
x_start, y_start = np.meshgrid(np.arange(0, Nx, N_batch), np.arange(0, Ny, N_batch))
x_start = x_start.flatten()
y_start = y_start.flatten()

x_start = np.array([40])
y_start = np.array([20])

# Params w.r.t. sampling positions
np.random.seed(103)
seeds = np.random.randint(1000, size = len(x_start))

# Iteration over batch
start_all = time.time()
war_dict = {}

for idx in range(len(x_start)):
    start = time.time()
    # Starting point for the current batch
    x, y = x_start[idx], y_start[idx]
    print('Batch @ (x, y) = ({}, {})'.format(x, y))
    
    # Select the sampling coordinates
    coords = batch_subsampling(N_batch, N_scan, seed = seeds[idx])
    print('Actual coverage = {}%'.format(round(100* coords.shape[0] / (N_batch**2), 2)))
    
    # Measurement data of the selected sampling positions w/ size = M x N_batch x N_batch
    formatter = SmartInspectDataFormatter(coords, 
                                          xmin = 0, xmax = N_batch, 
                                          ymin = 0, ymax = N_batch)
    data_sampled = formatter.get_data_matrix_pixelwise(data_roi[:, x + coords[:, 0], y + coords[:, 1]]) 
    
    # Kriging
    data_itp, war_list, V_dict = spatiotemporal_kriging(data_sampled, coords, n_lags, model = 'gaussian', 
                                                        maxlag = maxlag, ret_Vdict = True)
    war_dict.update({'x{}_y{}'.format(x, y) : war_list})
    

    # Data seving
    f_batch = 'x{}_y{}.npy'.format(x, y)
    save_data(data_sampled, 'npy_data/MUSE/{}/sampled'.format(f_date), f_batch)
    save_data(data_itp, 'npy_data/MUSE/{}/krigged'.format(f_date), f_batch)
    save_data(coords, 'npy_data/MUSE/{}/coords'.format(f_date), f_batch)
    save_data(V_dict['Variogram'], 'npy_data/MUSE/{}/Vari'.format(f_date), f_batch)
    del data_sampled, data_itp
    
    display_time(round(time.time() - start, 3))
    
# Computational time for entire ROI
print('#=====================#')
print('End of kriging for entire ROI')
display_time(round(time.time() - start_all, 3))

# save warning dictionary
with open('npy_data/MUSE/{}/war_dict.json'.format(f_date), 'w') as fp:
    json.dump(war_dict, fp)

#%% Reload each batch for visualization
# Base
data_sampled = np.zeros(data_roi.shape)
data_itp = np.zeros(data_roi.shape)
vari = np.zeros((M, n_lags, len(x_start)))
scan_map = np.zeros((Nx, Ny))

for idx in range(len(x_start)):
    x, y = x_start[idx], y_start[idx]
    f_batch = 'x{}_y{}.npy'.format(x, y)
    data_sampled[:, x : x + N_batch, y : y + N_batch] = np.load('npy_data/MUSE/{}/sampled/{}'.format(f_date, f_batch))
    data_itp[:, x : x + N_batch, y : y + N_batch] = np.load('npy_data/MUSE/{}/krigged/{}'.format(f_date, f_batch))
    vari[:, :, idx] = np.load('npy_data/MUSE/{}/Vari/{}'.format(f_date, f_batch))
    # Scan map
    p_scan = np.load('npy_data/MUSE/{}/coords/{}'.format(f_date, f_batch))
    scan_map[x + p_scan[:, 0], y + p_scan[:, 1]] = 1

err = data_roi - data_itp # Interpolation error

#%% Plots
z_axis = np.flip(np.arange(M))#* dz* 10**3 #[mm]
x_axis = np.arange(Nx)#* dx* 10**3 #[mm]
y_axis = np.arange(Ny)#* dx* 10**3 #[mm]
z_label = 'z'
x_label = 'x'
y_label = 'y'

# Value range
data_min = min(data_roi.min(), np.nanmin(data_itp))
data_max = max(data_roi.max(), np.nanmax(data_itp))
print('Actual data_min = {}, data_max = {}'.format(data_roi.min(), data_roi.max()))
print('data min = {}, data_max = {}'.format(data_min, data_max))


# Sampled data in the selected batch
fig_z = SliceFigure3D(data_sampled, 
                      0, 'Sampled data slice along z-axis', 
                      [z_axis, x_axis, np.flip(y_axis)], 
                      [z_label, x_label, y_label], 
                      0, False, display_text_info = True, info_height = 0.35,
                      data_min = data_min, data_max = data_max)


# Interpolated batch slice along z-axis
fig_z = SliceFigure3D(data_itp, 
                      0, 'Interpolated slice along z-axis', 
                      [z_axis, x_axis, np.flip(y_axis)],
                      [z_label, x_label, y_label], 
                      0, False, display_text_info = True, info_height = 0.35,
                      data_min = data_min, data_max = data_max)


# Actual batch slice along z-axis
fig_z = SliceFigure3D(data_roi, 
                      0, 'Actual data slice along z-axis', 
                      [z_axis, x_axis, np.flip(y_axis)],
                      [z_label, x_label, y_label], 
                      0, False, display_text_info = True, info_height = 0.35)#,
                      #data_min = data_min, data_max = data_max)

# =============================================================================
# # Slice along z-axis
# fig_z = SliceFigure3D(data_roi, 0, 'MUSE data slice along z-axis', 
#                       [z_axis, x_axis, y_axis], [z_label, x_label, y_label], 
#                       0, False, display_text_info = True, info_height = 0.35)
# =============================================================================


# Slice along y
# Sampled
fig_y = SliceFigure3D(data_sampled, 
                      2, 'Sampled data slice along y-axis', 
                      [z_axis, x_axis, y_axis], [z_label, x_label, y_label], 
                      0, False, display_text_info = True, info_height = 0.35,
                      data_min = data_min, data_max = data_max)

# Interpolated
fig_y = SliceFigure3D(data_itp, 
                      2, 'Interpolated data slice along y-axis', 
                      [z_axis, x_axis, y_axis], [z_label, x_label, y_label], 
                      0, False, display_text_info = True, info_height = 0.35,
                      data_min = data_min, data_max = data_max)

# Actual
fig_y = SliceFigure3D(data_roi, 
                      2, 'Actual data slice along y-axis', 
                      [z_axis, x_axis, y_axis], [z_label, x_label, y_label], 
                      0, False, display_text_info = True, info_height = 0.35,
                      data_min = data_min, data_max = data_max)

# Variogram
bins = V_dict['bins']
fig_y = SliceFigure3D(vari[:, :-1, :], 
                      2, 'Variogram', 
                      [z_axis, bins[:-1], np.arange(len(x_start))], [z_label, 'distance', 'Batch No.'], 
                      0, False, display_text_info = True, info_height = 0.35,
                      data_min = None, data_max = None)


# Interpolation error
# =============================================================================
# fig_y = SliceFigure3D(err, 
#                       2, 'Interpolation error along y-axis', 
#                       [z_axis, x_axis, y_axis], [z_label, x_label, y_label], 
#                       0, False, display_text_info = True, info_height = 0.35)
# =============================================================================

plt.figure()
plt.imshow(scan_map.T)
plt.colorbar()
plt.title('Spatial subsampling: selected positions')
plt.xlabel('x/dx')
plt.ylabel('y/dy')


plt.show()



