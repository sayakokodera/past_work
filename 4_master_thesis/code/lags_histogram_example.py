#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Frequency Kriging
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as scdis

from spatial_subsampling import batch_subsampling
from spatial_subsampling import get_all_grid_points
from spatial_lag_handling import SpatialLagHandlingScalar

from frequency_variogram import FrequencyVariogramRaw
from frequency_variogram import FrequencyVariogramDNN

#%% Functions
def compute_histogram(bins, slh):
    """
    Compute histogram w.r.t. the given bins including BOTH edges
    """
    # Get the available lags
    lags_avail = slh.get_valid_lags() 
    
    hist = np.zeros(bins.shape)
    for idx, item in enumerate(bins):
        if item in lags_avail:
            hist[idx] = len(slh.get_indices(item))
        else:
            print('Current lag {} is not found!'.format(item))
    return hist
        


#%% Parameters
dx = round(0.5* 10**-3, 5)
N_batch = 10
maxlag = round(dx* 5* np.sqrt(2), 10) + 10**-10

#%% Full grid points
p_full = get_all_grid_points(N_batch, N_batch) # uniteless
s_full = np.around(dx* p_full, 10) # [m]
lags_raw_full = np.around(scdis.pdist(s_full), 10) #[m]

# SLH class
slh_full = SpatialLagHandlingScalar(lags_raw_full, maxlag)
# Base lags = bins for histogram
lags = slh_full.get_valid_lags() # WITHOUT 0!!!!

# histogram
hist_full = compute_histogram(lags, slh_full)
total_full = np.sum(hist_full)

#%% Sampled positions: 80% coverage
p_smp1 = batch_subsampling(N_batch, 80)
s_smp1 = np.around(dx* p_smp1, 10) #[m]
lags_raw_smp1 = np.around(scdis.pdist(s_smp1), 10) #[m]

# SLH class
slh_smp1 = SpatialLagHandlingScalar(lags_raw_smp1, maxlag)

# histogram
hist_smp1 = compute_histogram(lags, slh_smp1)
total_smp1 = np.sum(hist_smp1)

#%% Sampled positions: 30% coverage
p_smp2 = batch_subsampling(N_batch, 20)
s_smp2 = np.around(dx* p_smp2, 10) #[m]
lags_raw_smp2 = np.around(scdis.pdist(s_smp2), 10) #[m]

# SLH class
slh_smp2 = SpatialLagHandlingScalar(lags_raw_smp2, maxlag)

# histogram
hist_smp2 = compute_histogram(lags, slh_smp2)
total_smp2 = np.sum(hist_smp2)

#%% Plots
plt.plot(lags, hist_smp1 / hist_full, label = 'r_smp1, 80%')
plt.plot(lags, hist_smp2 / hist_full, label = 'r_smp2, 20%')
plt.plot(lags, hist_full / hist_full, label = 'r_full')
plt.legend()
plt.title('Relative histogram')
plt.xlabel('lags')


#%% Scalar-valued FV DNN 
cfv1 = FrequencyVariogramDNN(dx, N_batch, maxlag = maxlag)
cfv1.set_positions(s_smp1)
cfv1.set_data()

