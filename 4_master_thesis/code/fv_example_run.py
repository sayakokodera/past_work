#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Frequency Kriging
"""
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
import time

from tools.display_time import display_time

from spatial_subsampling import batch_subsampling
from signal_denoising import denoise_svd
from defects_synthesizer import DefectsSynthesizer
from phase_shifter import PhaseShifter

from frequency_variogram import FrequencyVariogramRaw

plt.close('all') 


"""
Notations:
    a, A: data
    b, B: pair-wise difference of data (= i.e. incremental process)
    v, V: variance
    
    _t: data/covariance in time domain
    _s: data/covariance in space domain
    _f: data/covariance in frequency domain
    _a: data/covariance in angular domain
    
    _st: data/covariance in space-time domain
    _at: data/covariance in angulat-time domain
    _sf: data/covariance in space-frequency domain
    _af: data/covariance in angular-frequency domain
    
    p_: scan positions as indices, i.e. unitless
    s_: scan positions in [m]
    d_: spartial lan corresponds to the indices (p) (unitless)
        => using the lag based on the indices prevents the fractional error which causes to count 
        the same lags as two different ones
    h_: spatial lag in [m], h_ij = sqrt((s_i - s_j)**2)
     
"""


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


def get_mean_power(sig):
    """
    Parameters
    ----------
        sig: np.array()
            Multiple signals with axis = 0 being the axis of each signal
    """
    return np.mean(np.sum(sig**2, axis = 0))
    
def print_snr(noise, a_clean):
    Pn = get_mean_power(noise) # Noise power within the ROI. i.e. dt* M
    Ps_actual = get_mean_power(a_clean) # Actual mean value of the signal energy per A-Scan
    print('*** SNR = {}dB ***'.format(10* np.log10(Ps_actual / Pn)))
    print('Ps_actual = {}'.format(Ps_actual))
    print('Pn_actual = {}'.format(Pn))


#%% Parameters for FK
"""
Random factors here:
    defect positions
    defect reflectivity
    noise
    coverage
    scan positions
"""
### Specimen setting ###
# ROI and its location
N_batch = 10 # = Nx = Ny
M = 512 # Temporal dimension of ROI (= excluding the time offset)
Nt_offset = 391
Nt_full = Nt_offset + M
dx = 0.5* 10**-3 #[m]

# AGWN
seedNo_noise = 92#np.random.randint(10**5)
Pn_dt1 = 0.005#0.75*10**-2 # Noise power / dt 
Pn_dt2 = 10**-5

### Pulse setting ###
# Phase
phi1 = 0 # [degree], Phase
phi2 = 360 # [degree], Phase

### Spatial subsampling setting ###
#np.random.seed()
#coverage = np.random.uniform(0.02, 0.3)
N_scan = 16#int(coverage* N_fine**2)
seedNo_ss = 353#np.random.randint(10**5) #277

### Position to predict, should be in an array form for scdis.cdist ###
p0 = np.array([[5, 5]])

### FV/FK setting ###
# FV parameters
f_cutoff = None #[fmin, fmax], corresponds to bin (i.e. index), None if we don't want to limit the freq. range
maxlag = 0.5* np.sqrt(2)* N_batch* dx #+ 0.1* dx_coarse# [m]
use_smp = False # True, if FV to be calculated using only the samples
smooth = True
deg = 5
# FK parameters: Tikhonov regularization for weights calculation
reg_Tik = True
alpha_fac = 0.01*10**3 # Regularization factor

#%% Scan position setting
# Scan positions = all grid points
xx, yy = np.meshgrid(np.arange(N_batch), np.arange(N_batch)) # Scan grids
p_full = np.array([xx.flatten(), yy.flatten()]).T #  Scans are taken @ all grid points as indices, unitless
s_full = np.around(dx* p_full, 10)

    
#%% Clean data generation

# from ultrasonic_imaging_python.forward_models.propagation_models_3D_single_medium import PropagationModel3DSingleMedium
# from ultrasonic_imaging_python.forward_models.data_synthesizers import DataSynthesizer

# from ultrasonic_imaging_python.definitions import units
# ureg = units.ureg

# # Variables
# # Defect position
# Ndefect = 2#np.random.randint(10)
# amp_low = 0.4
# seedNo_def = 267#np.random.randint(10**5) #431
# seedNo_amp = 100

# # Constant parameters
# # Measurement parameters
# c0 = 5900 #[m/S]
# fS = 80* 10**6 #[Hz]
# dt = 1/fS #[S]
# dz = 0.5* c0* dt #[m]
# # Pulse setting
# l_pulse = 238
# fC = 3.36* 10**6 #[Hz]


# # FWM setting
# spc_param = {# Specimen parameters, dimension here = ROI!!!!!!
#     'dxdata': dx * ureg.meter,
#     'dydata': dx * ureg.meter,
#     'c0': c0 * ureg.meter / ureg.second,
#     'fS': fS * ureg.hertz,
#     'Nxdata': N_batch,
#     'Nydata': N_batch,
#     'Ntdata': M, # = Nt_full - Nt_offset!!!!!
#     'Nxreco': N_batch,
#     'Nyreco': N_batch,
#     'Nzreco': M, # = Nt_full - Nt_offset!!!!!
#     'anglex': 0 * ureg.degree,
#     'angley': 0 * ureg.degree,
#     'zImage': -Nt_offset * dz * ureg.meter,
#     'xImage': 0 * 0.5 * 10**-3 * ureg.meter,
#     'yImage': 0 * 0.5 * 10**-3 * ureg.meter,
#     't0': Nt_offset* dt * ureg.second
#     }
# apd_param = {# Apodization parameters
#     'max_angle': 25 * ureg.degree
#     }
# pulse_params = {# Pulse parameters
#     'pulseLength': l_pulse,
#     'fCarrier': fC * ureg.hertz,
#     'B': 0.3,
#     'fS': fS * ureg.hertz
#     }
# # FWM initialization
# model = PropagationModel3DSingleMedium()
# model.set_parameters(spc_param)
# model.set_apodizationmodel('Bartlett', apd_param)
# model.set_pulsemodel('Gaussian', pulse_params)


# # Defect positions: ROI
# defect_dict = {
#     'x': np.arange(N_batch),
#     'y': np.arange(N_batch),
#     'z': np.arange(int(0.5* l_pulse), int(M - 0.5* l_pulse)),
#     'amp_low' : amp_low
#     }
# ds = DefectsSynthesizer(M, N_batch, N_batch, defect_dict)
# ds.set_defects(Ndefect, seed_p = seedNo_def, seed_amp = seedNo_amp)
# defmap = ds.get_defect_map_3d() # shape = M x Nx x Ny


# # Data synthesization (in space-time domain)
# synth = DataSynthesizer(model)#, pulse_model = model._pulse_model)
# a_st_clean_raw = synth.get_data(defmap) # shape = M x 1 x 1 x 1 x 1 x Nxdata x Nydata
# a_st_clean_raw = np.reshape(a_st_clean_raw.flatten('F'), defmap.shape, 'F') # shape = M x Nxdata x Nydata

#%% Load the existing data
from tools.npy_file_writer import num2string

# file settings
Nt_offset = 391
Ndefect = 2
dataNo = 0
path = 'npy_data/ML/train/clean_data/depth_{}/Ndef_{}'.format(Nt_offset, num2string(Ndefect))
fname = '{}.npy'.format(num2string(dataNo))
# Load
a_st_clean_roi = np.load('{}/{}'.format(path, fname))

# Choose the batch
x_start, y_start = 0, 15
a_st_clean_batch = a_st_clean_roi[:, x_start: x_start + N_batch, y_start : y_start + N_batch]

#%% Data modification
# Normalize the clean data
vmax = np.abs(a_st_clean_batch).max()
a_st_clean_batch = np.nan_to_num(a_st_clean_batch / vmax) # In case the batch only contains zero: NaN -> 0

# Phase shift
ps = PhaseShifter()
a_st_clean1 = ps.shift_phase(a_st_clean_batch, phi1, axis = 0)
a_st_clean2 = ps.shift_phase(a_st_clean_batch, phi2, axis = 0)
print('phi = {}'.format(ps.phi))

# AGWN
# Noise 1
np.random.seed(seedNo_noise)
noise1 = np.random.normal(scale = np.sqrt(Pn_dt1), size = a_st_clean_batch.shape) # shape = M x Nx x Ny
#noise1 = np.random.uniform(low = -np.sqrt(Pn_dt1), high = np.sqrt(Pn_dt1), size = a_st_clean.shape) # shape = M x Nx x Ny
# Noise 2
np.random.seed(seedNo_noise)
noise2 = np.random.normal(scale = np.sqrt(Pn_dt2), size = a_st_clean_batch.shape) # shape = M x Nx x Ny
#noise2 = np.random.uniform(low = -np.sqrt(Pn_dt2), high = np.sqrt(Pn_dt2), size = a_st_clean.shape) # shape = M x Nx x Ny

# Noisy data
a1 = a_st_clean1 + noise1
a2 = a_st_clean2 + noise2

# Normalize
a1 = a1 / np.abs(a1).max()
a2 = a2 / np.abs(a2).max()


print('max(a_st_clean_batch) = {}'.format(np.abs(a_st_clean_batch).max()))
print_snr(noise1, a_st_clean1)
print_snr(noise2, a_st_clean2)


#%% FV of the entire batch

### For clean signal ###
cfv1 = FrequencyVariogramRaw(grid_spacing = dx, maxlag = maxlag)
cfv1.set_positions(s_full)
cfv1.set_data(a1, M, f_range = None)
cfv1.compute_fv()
cfv1.smooth_fv(deg, ret_normalized = True)
fv1_norm = cfv1.get_fv()
fv1 = cfv1.denormalize_fv(fv1_norm)

lags_full = cfv1.get_lags()

### For noisy signal ###
cfv2 = FrequencyVariogramRaw(grid_spacing = dx, maxlag = maxlag)
cfv2.set_positions(s_full)
cfv2.set_data(a2, M, f_range = None)
cfv2.compute_fv()
cfv2.smooth_fv(deg, ret_normalized = True)
fv2_norm = cfv2.get_fv()
fv2 = cfv2.denormalize_fv(fv2_norm)

# Difference b/w clean & noisy FV
fvnorm_diff = np.abs(fv1_norm - fv2_norm)
fv_diff = np.abs(fv1 - fv2)

print('max(fvnorm_diff) for relvant fbin = {}'.format(fvnorm_diff[18:25, :].max()))


#%% Plots

plt.figure()
plt.plot(a1[:, p0[0, 0], p0[0, 1]], label = 'a1')
plt.plot(a2[:, p0[0, 0], p0[0, 1]], label = 'a2')
plt.legend()

plt.figure()
plt.plot(lags_full, fv1_norm[18:25, :].T)
plt.title('Normalized FV1 (freq. bin [18, 25])')
plt.xlabel('lag')


plt.figure()
plt.plot(lags_full, fv2_norm[18:25, :].T)
plt.title('Normalized FV2 (freq. bin [18, 25])')
plt.xlabel('lag')


plt.figure()
plt.plot(lags_full, fvnorm_diff[18:25, :].T)
plt.title('Normalized FV difference (freq. bin [18, 25])')
plt.xlabel('lag')

plt.figure()
plt.plot(lags_full, fv1_norm[-50:, :].T)
plt.title('Normalized FV1: only noise (freq. bin [-50, -1])')
plt.xlabel('lag')


#%% 3D plots
from ultrasonic_imaging_python.visualization.slice_figures import SliceFigure3D

Nx, Ny = 30, 30

z_axis = np.flip(np.arange(M))#* dz* 10**3 #[mm]
x_axis = np.arange(Nx)#* dx* 10**3 #[mm]
y_axis = np.arange(Ny)#* dx* 10**3 #[mm]
z_label = 'z'
x_label = 'x'
y_label = 'y'

fig_y = SliceFigure3D(a_st_clean_roi, 
                      2, 'Clean ROI data along y-axis', 
                      axis_range = [z_axis, x_axis, y_axis], 
                      axis_label = [z_label, x_label, y_label], 
                      initial_slice = 0, display_text_info = True, info_height = 0.35
                      )

x_axis = np.arange(N_batch)#* dx* 10**3 #[mm]
y_axis = np.arange(N_batch)#* dx* 10**3 #[mm]
fig_y = SliceFigure3D(a_st_clean_batch, 
                      2, 'Clean batch data along y-axis', 
                      axis_range = [z_axis, x_axis, y_axis], 
                      axis_label = [z_label, x_label, y_label], 
                      initial_slice = 0, display_text_info = True, info_height = 0.35
                      )



    
    

