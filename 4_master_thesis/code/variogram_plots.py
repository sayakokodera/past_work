#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 02:35:33 2021

@author: sayakokodera
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio

import skgstat as skg

from spatial_subsampling import get_all_grid_points


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

# Select the ROI
A_roi = muse_data['data'][zmin:zmax, xmin:xmax, ymin:ymax]
A_roi = A_roi / np.abs(A_roi).max() # normalize
del muse_data['data']


#%% Variogram

N_batch = 20
p_full = get_all_grid_points(N_batch, N_batch)
maxlag = np.around(N_batch/2* np.sqrt(2), 10) # in [m]
n_lags = N_batch

x_start = 10#max(0, np.random.randint(Nx - N_batch)) #21
y_start = 20
z = 300

batch = A_roi[z, x_start+p_full[:, 0], y_start+p_full[:, 1]]

# Initialize the variogram class
V = skg.Variogram(p_full, batch, n_lags = n_lags,  model = 'gaussian', maxlag = maxlag)
bins = V.bins
lags = np.concatenate((np.zeros(1), bins))
models = ['spherical', 'exponential', 'matern', 'tent']

vari = np.zeros((len(lags), 4))

# For plots
fig, _a = plt.subplots(2,3, figsize=(18, 10), sharex=True, sharey=True)
axes = _a.flatten()

for idx, model in enumerate(models[:-1]):
    V.model = model
    V.plot(axes=axes[idx], hist=False, show=False)
    vari[:, idx] = V.transform(lags)
    if model == 'exponential':
        vari[:, idx] = 7/6* vari[:, idx]

# Tent model
vari[:, 3] = 0.006* lags**1.2 # tent model

# Scale
vari = vari* 10

# Covariance
c_matern = vari[:, 2].max() - vari[:, 2]

# Plot
plt.figure()
plt.plot(lags, vari[:, 0], label = models[0])
plt.plot(lags, vari[:, 1], label = models[1])
plt.plot(lags, vari[:, 2], label = models[2])
plt.plot(lags, vari[:, 3], label = models[3])
plt.legend()


import sys
sys.exit()

#%% Save
import tools.tex_1D_visualization as pgf1d

path_tex = '/Users/sayakokodera/Uni/Master/MA/tex/MA/figures/coords_1D'
colors = ['tui_blue', 'tui_orange', 'fri_green', 'tui_red']
linestyle = ['very thick', 'very thick', 'very thick', 'very thick']

# Save: variograms
pgf1d.generate_coordinates_for_addplot(lags, '{}/variograms.tex'.format(path_tex), False, colors, linestyle,
                                       vari[:, 0], vari[:, 1], vari[:, 2], vari[:, 3])
                                       
# Save: cov vs vari
pgf1d.generate_coordinates_for_addplot(lags, '{}/cov_vs_vari.tex'.format(path_tex), False, colors[:2], 
                                       linestyle[:2], vari[:, 2], c_matern)




