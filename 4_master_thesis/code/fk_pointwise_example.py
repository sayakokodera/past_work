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

import scipy.io as scio
import scipy.signal as scsi
import numpy.fft as fft
import scipy.spatial.distance as scdis
import time
from numpy.polynomial import Polynomial

from tools.display_time import display_time

from smart_inspect_data_formatter import SmartInspectDataFormatter
from spatial_subsampling import batch_subsampling
from signal_denoising import denoise_svd
from signal_denoising import denoise_lowpass
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
    


def pairwise_difference(val): # Done in FrequencyVariogram
    """
    Input
    -----------
        val: a vector (int, float, complex) with sieze = N
        
    Output
    ------
        pdiff: a vector with size = N* (N-1)/2
            pair-wise differences
    """
    N = len(val)
    arr1 = np.repeat(val.reshape(N, 1), N-1, axis = 1)
    arr2 = np.repeat(val[:-1].reshape(N-1, 1), N, axis = 1).T
    # Pair-wise difference = arr1 - arr2
    diff = arr1 - arr2
    diff[np.triu_indices_from(diff)] = np.nan + 1j* np.nan # Replace the upper diagonals (i.e.repeats) with NaNs
    # Remove NaNs
    pdiff = diff.flatten('F')
    pdiff = pdiff[~np.isnan(pdiff)] 
    return pdiff
    

def limit_lag(lag, maxlag):
    """ Limit the lag to the given range
    Input
    -----
        lag : np.ndarray
            Spatial lags
        maxlag : float
            Maximum spatial lag to limit the range

    Output
    -------
        indicies: np.ndarray, size = N
            Indicies corresponding to the lags which are smaller than maxlag
        lag_valid: np.ndarray, size = N
            Lags within the given range
            

    """ 
    # Consider only the elements whose lag is smaller than maxlag
    indices = np.where(lag <= maxlag)[0]
    # Lags limited within the range
    lag_valid = lag[indices]
    
    return indices, lag_valid


def incremental_process(p, A_sf): # Done in FrequencyVariogram
    """ Calculate the space-freq. spectra fo teh incremental process (i.e. pair-wise difference) of the sampled data
    
    Input
    -----
        p: np.ndarray(K, 2) with K = # of scan positions
            Scan positions
        A_sf: np.ndarray(Nf, K), complex
            Space-freq. spectra of the scampled data
    
    Output
    ------
        h_raw: np.ndarray(L) with L = K*(K-1)/2, float
            Pair-wise spatial lag
        B_sf: np.ndarray(Nf, L) with L = K*(K-1)/2, complex
            Space-freq. spectra of the incremental process of the sampled data
            B_sf[:, l] = A_sf[:, i] - A_sf[:, j] with l = index of h_raw for ||s_i - s_j ||_2
        
    """
    Nf = A_sf.shape[0]
    h_raw = scdis.pdist(p) # Pair-wise distance of sampled positions
    
    # Space-freq. spectrum 
    B_sf = 1j* np.zeros((Nf, len(h_raw)))
    for row in range(Nf):
        B_sf[row, :] = pairwise_difference(A_sf[row, :])
    
    return h_raw, B_sf
        
    
def experimental_frequency_variogram(h_raw, B_sf):
    """ Calculate the variance of the space-freq. spectra for the incremental process of the sampled data
    (= experimental frequency variogram(FV)?) shown in Eq(8) in [1]
    
    Input
    -----
        h_raw: np.ndarray(L) with L = K*(K-1)/2, float
            Pair-wise spatial lag
        B_sf: np.ndarray(Nf, L) with L = K*(K-1)/2, complex
            Space-freq. spectra of the incremental process of the sampled data
            B_sf[:, l] = A_sf[:, i] - A_sf[:, j] with l = index of h_raw for ||s_i - s_j ||_2
    
    Output
    ------
        h: np.ndarray(Nh), float
            Pair-wise spatial lag
        V_sf: np.ndarray(Nf, Nh), real AND positive values
            Variance of the space-freq. spectra for the given incremental process 
            
    Reference
    ---------
        [1] T. Subba Rau et al, 2017, "ON THE FREQUENCY VARIOGRAM AND ON FREQUENCY DOMAIN METHODS FOR THE ANALYSIS OF 
        SPATIO-TEMPORAL DATA"
    
    """
    Nf = B_sf.shape[0]
    h = np.concatenate((np.zeros(1), np.unique(h_raw)))
    
    # Experimental FV = expected value of space-freq. periodogram of the incremental process for a fixed sptial lag
    Vb_sf = np.zeros((Nf, len(h)))
    # Exp. FV calculation for each (fixed) spatial lag
    for col, curr_dist in enumerate(h):
        if col == 0:
            pass
        else:
            idx = np.flatnonzero(h_raw == curr_dist) # Indices corresponding to the current lag
            Rbb = B_sf[:, idx].conj()* B_sf[:, idx]
            #mean = np.mean(B_sf[:, idx])
            #print('h = {}, mean(spectrum) = {}'.format(col, mean))
            #Rbb = (B_sf[:, idx] - mean).conj()* (B_sf[:, idx] - mean) unnecessary because the mean is almost 0
            # Check if Rbb is real and positive
            if len(np.where(np.isreal(Rbb).astype(int) == 0)[0]) != 0:        
                raise ValueError('Rbb is NOT real with lag No.{}'.format(col))
                #print('Rbb is real with lag = {}'.format(curr_dist))
            elif len(np.where(Rbb < 0)[0]) != 0:
                raise ValueError('Rbb is NOT positive with lag No.{}'.format(col))
            
            Vb_sf[:, col] = np.mean(Rbb, axis = 1) # Mean over the lag
            
    # Check if Vb_sf is real and positive semi-definite
    if len(np.where(np.isreal(Vb_sf).astype(int) == 0)[0]) != 0:        
        raise ValueError('Experimental FV is NOT real')
    elif len(np.where(Vb_sf < 0)[0]) != 0:
        raise ValueError('Experimental FV is NOT positive semidefinite!')
    
    return h, Vb_sf


def smoothed_frequency_variogram(lag, fv_raw, deg):
    """ Smooth the experimental frequency variogram via curve(polynomial) fitting. FV exhibits the different "shape"
    depending on the frequency, thus we compute a polynomial for each frequency component. 

    Input
    -----
    lag : np.ndarray(Nlag), float
        Spatial lag 
    fv_raw : np.ndarray(Nf, Nlag), real & positive
        Experimental variogram 
    deg : int, positive
        Polynomial degree

    Output
    -------
    fv_smt : np.ndarray(Nf, Nlag), real & positive
        Smoothed frequency variogram 
        For a fixed freq. bin and a fixed lag, smoothed_fv is calculated as 
            smoothed_fv[f_bin, i] = c0 + c1* lag[i] + c2* lag[i]**2 + c3* lag[i]**3 + .....
        -> Meaning, for all spatial lags:
            smoothed_fv[f_bin, :] = np.dot(mat_lag, coeff):
                mat_lag = np.array(Nlag, deg+1)
                        = np.array([
                            [1, lag[0], lag[0]**2, ..... lag[0]**deg],
                            [1, lag[1], lag[1]**2, ..... lag[1]**deg],
                            [1, lag[2], lag[2]**2, ..... lag[2]**deg],
                            ...
                        ])
                coeff = np.array([c0, c1, c2, ....c_deg])
    """
    # Construct mat_lag
    mat_lag = np.zeros((len(lag), deg+1))
    for col in range(mat_lag.shape[1]):
        mat_lag[:, col] = lag**col
    
    # Base
    fv_smt = np.zeros((fv_raw.shape))
    
    # Iterate over freq. bins
    for f_bin in range(fv_smt.shape[0]):
        poly = Polynomial.fit(lag, fv_raw[f_bin, :], deg = deg)
        coeff = poly.convert().coef
        fv_smt[f_bin, :] = np.dot(mat_lag, coeff)
    
    return fv_smt



def plot_2D_Gaussian(hist, valmax, title, xlabel, ylabel):
    # Axis setting
    xx, yy = np.meshgrid(np.arange(hist.shape[0]), np.linspace(-valmax, valmax, hist.shape[1]))
    # Plot
    fig = plt.figure(figsize=(13, 7))
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(xx, yy, hist.T, rstride=1, cstride=1, cmap='coolwarm', edgecolor='none')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel('Histogram')
    ax.set_title(title)
    fig.colorbar(surf, shrink=0.5, aspect=5) # add color bar indicating the PDF
    ax.view_init(60, 35)
    

def check_symmetric(mat, rtol=1e-05, atol=1e-08):
    return np.allclose(mat, mat.T, rtol=rtol, atol=atol)


def check_positive_definiteness(mat, rtol=1e-05, atol=1e-08):
    # Check Finv is real, symmetric positive definite matrix
    if len(np.where(np.isreal(mat).astype(int) == 0)[0]) != 0:        
        raise ValueError('Given matrix is not real!')
    elif check_symmetric(mat) == False:
        raise ValueError('Given matrix  is not symmetric!')
    elif np.any(np.linalg.eigvals(mat)) < 0: 
        raise ValueError('Given matrix is not positive-definite!')
        
        
def kriging_weights_sf_domain(F, g0hat, reg_Tik = False, alpha = None):
    """ Compute the Kriging weights for space-freq. domain
    
    Input
    -----
    F : np.ndarray(K, K), real
        Raw freq. variogram matrix for a SINGLE freq. bin.
        F should be positive-semidefinite (also means F.T == F)
        Here, K = # of scans to considers for prediction.
    gohat : np.ndarray(K), real, positive
        Estimated freq. variances b/w s0 and the K neighboring scan positions for a SINGLE freq. bin
    reg_Tik: boolean (False by default)
        True, if the weights should be calculated using Tikhonov regularization (i.e. make F nonsingular) to avoid 
        weights "explosion"
    alpha: float
        Tikhonov factor which should be given, when reg_Tik == True

    Output
    ------
    w: np.ndarray(1, K), real
        Space-freq. Kriging weights for the selected freq. bin

    """
    # Without Tikhonov regularization: use Finv directly
    if reg_Tik == False:
        Finv = np.linalg.inv(F)
        w = np.dot(g0hat.reshape(1, len(g0hat)), Finv) # Weights
    # With Tikhonov regularization
    else:
        if alpha is None:
            raise AttributeError('FK weights: Tikhonov factor is not given!')
        X = np.linalg.inv(np.dot(F, F) + alpha* np.eye(F.shape[0]))
        w = np.dot(X, np.dot(F, g0hat))
        w = np.reshape(w, (1, len(g0hat))) # Convert w into a row vector
        
    return w
    
def print_snr(Pn_dt, a_clean):
    Pn = Pn_dt* a_clean.shape[0] # Noise power within the ROI. i.e. dt* M
    Ps_actual = np.mean(np.sum(a_clean**2, axis = 0)) # Actual mean value of the signal energy per A-Scan
    print('*** SNR = {}dB ***'.format(10* np.log10(Ps_actual / Pn)))



#%% Parameters for FK

### Specimen setting ###
# ROI and its location
M = 512 # Temporal dimension of ROI (= excluding the time offset)
Nt_full = 1800
# Defect position
Ndefect = 7#np.random.randint(10)
amp_low = 0.4
seedNo_def = 267#np.random.randint(10**5) #431
seedNo_amp = 743
# AGWN
Pn_dt = 10**-4 # Noise power / dt
snr_dB = 15 # [dB]
seedNo_noise = np.random.randint(10**5) #92

### Pulse setting ###
# Phase
phi = 1.25* np.pi # Phase
# Pulse length
l_pulse_base = 128 # Base pulse length 
fluct_pulse = 10 # Fluctuation of pulse length
seedNo_pulse = 0#np.random.randint(10**5)
# Center frequency fC
fluct_fC = 0.1 # Relative fluctuation 
seedNo_fC = 0#np.random.randint(10**5)

### Sampling setting ###
# Sampling vs interpolation grid
grid_ratio = 1 # = dx_fine / dx
dx_coarse = round(0.5*10**-3, 5) #[m]
dx_fine = round(0.5*10**-3, 5) #[m], fine grding for calculating the field statistics (i.e. variogram)
N_batch = 10
#N_coarse = int(N_fine/grid_ratio) 
# Parameters for spatial subsampling
N_ss = N_batch # For specifying the subsampling region
np.random.seed()
coverage = np.random.uniform(0.02, 0.3)
N_scan = 35#int(coverage* N_fine**2) #16
seedNo_ss = 97557#np.random.randint(10**5) #97557


### Position to predict, should be in an array form for scdis.cdist ###
s0 = np.array([[0.5* N_batch* dx_coarse, 0.5* N_batch* dx_coarse]])  # [m]
p0_coarse = (s0/dx_coarse).astype(int)
p0_fine = (s0/dx_fine).astype(int)


### FV/FK setting ###
# FV parameters
f_cutoff = None #[fmin, fmax], corresponds to bin (i.e. index), None if we don't want to limit the freq. range
maxlag = 0.5* np.sqrt(2)* N_batch* dx_coarse + 10**-9 # [m]
use_smp = False # True, if FV to be calculated using only the samples
smooth = True
deg = 5
# FK parameters: Tikhonov regularization for weights calculation
reg_Tik = True
alpha_fac = 0.01*10**3 # Regularization factor
    
#%% Clean data generation
# from ultrasonic_imaging_python.forward_models.propagation_models_3D_single_medium import PropagationModel3DSingleMedium
# from ultrasonic_imaging_python.forward_models.data_synthesizers import DataSynthesizer

# from ultrasonic_imaging_python.definitions import units
# ureg = units.ureg

# #Parameter Setting 
# Dimensions
Nx = N_batch
Ny = N_batch
Nt_offset = Nt_full - M

# Measurement parameters
c0 = 5900 #[m/S]
fS = 80* 10**6 #[Hz]
dt = 1/fS #[S]
dz = 0.5* c0* dt #[m]

# # Pulse parameters
# # Variational
# np.random.seed(seedNo_pulse)
# l_pulse = l_pulse_base# + np.random.randint(-fluct_pulse, fluct_pulse)
# print(l_pulse)
# # Fixed
# np.random.seed(seedNo_fC)
# fC = 3.36* 10**6#* (1 + np.random.uniform(-fluct_fC, fluct_fC)) #[Hz]
# print(fC* 10**-6)

# # FWM setting
# spc_param = {# Specimen parameters, dimension here = ROI!!!!!!
#     'dxdata': dx_fine * ureg.meter,
#     'dydata': dx_fine * ureg.meter,
#     'c0': c0 * ureg.meter / ureg.second,
#     'fS': fS * ureg.hertz,
#     'Nxdata': Nx,
#     'Nydata': Ny,
#     'Ntdata': M, # = Nt_full - Nt_offset!!!!!
#     'Nxreco': Nx,
#     'Nyreco': Ny,
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
#     'x': np.arange(Nx),
#     'y': np.arange(Ny),
#     'z': np.arange(int(0.5* l_pulse), int(M - 0.5* l_pulse)),
#     'amp_low' : amp_low
#     }
# ds = DefectsSynthesizer(M, Nx, Ny, defect_dict)
# ds.set_defects(Ndefect, seed_p = seedNo_def, seed_amp = seedNo_amp)
# defmap = ds.get_defect_map_3d() # shape = M x Nx x Ny


# # Data synthesization (in space-time domain)
# synth = DataSynthesizer(model)#, pulse_model = model._pulse_model)
# a_st_clean_raw = synth.get_data(defmap) # shape = M x 1 x 1 x 1 x 1 x Nxdata x Nydata
# a_st_clean_raw = np.reshape(a_st_clean_raw.flatten('F'), defmap.shape, 'F') # shape = M x Nxdata x Nydata

#%% Use the existing data instead of synthesizing

from tools.npy_file_writer import num2string

# file settings
Nt_offset = 391
Ndefect = 2
dataNo = 173
path = 'npy_data/ML/training/clean_data/depth_{}/Ndef_{}'.format(Nt_offset, num2string(Ndefect))
fname = '{}.npy'.format(num2string(dataNo))
# Load
a_st_clean_roi = np.load('{}/{}'.format(path, fname))

# Choose the batch
x_start, y_start = 5, 0
a_st_clean_batch = a_st_clean_roi[:, x_start: x_start + N_batch, y_start : y_start + N_batch]
a_st_clean_raw = np.copy(a_st_clean_batch)


#%% Data modification
# Phase shift
ps = PhaseShifter()
a_st_clean = ps.shift_phase(a_st_clean_raw, phi, axis = 0)
#a_st_clean = a_st_clean / np.abs(a_st_clean).max()

# AGWN
Pn = Pn_dt* a_st_clean.shape[0] # Noise power within the ROI. i.e. dt* M
np.random.seed(seedNo_noise)
noise = np.random.normal(scale = np.sqrt(Pn_dt), size = a_st_clean.shape) # shape = M x Nx x Ny

# Scaling for adjusting the signal amplitude according to the given SNR
# Ps_target = Pn* (10**(0.1*snr_dB)) # Target signal power according to the given SNR
#Ps_actual = np.mean(np.sum(a_st_clean**2, axis = 0)) # Actual mean value of the signal energy per A-Scan
# if Ps_actual > 0.0:
#     a_st_clean = np.sqrt(Ps_target / Ps_actual)* a_st_clean
# else:
#     a_st_clean = a_st_clean

# Noisy signal
a_st_noisy = a_st_clean + noise
vmax = np.amax(np.abs(a_st_noisy))
a_st = np.copy(a_st_noisy)/vmax

# SNR
print_snr(Pn_dt, a_st_clean)

#%% Spectral analysis of the entire batch

# Space-freq. domain
freq = 10**-6* fft.rfftfreq(M) # = [0, f_Ny], f_Ny is included!, normalized with fS!!!
A_sf = fft.rfft(a_st, n = M, axis = 0) # in space-freq. domain, i.e. preserving the data format
A_f = np.sum(A_sf, axis = (1, 2)) # spectra in freq. domain

# Limit the freq. range
if f_cutoff == None: # i.e. not limiting the range
    fmin = 0
    fmax = len(freq)
else:
    fmin = f_cutoff[0]
    fmax = f_cutoff[1]
f_interest = np.arange(fmin, fmax)

# Incremental process (b_st, B_sf)
# All scan positions
xx, yy = np.meshgrid(np.arange(N_batch), np.arange(N_batch)) # Scan grids
p_full = np.array([xx.flatten(), yy.flatten()]).T #  Scans are taken @ all grid points as indices, unitless

# Raw lags & space-freq. spectra of the incremental process
d_raw, B_sf_raw = incremental_process(p_full, np.reshape(A_sf, (len(freq), N_batch**2), 'F'))
h_raw = dx_fine* d_raw

# Limit the spatial lags to maxlag
idx_valid, h_raw_valid = limit_lag(h_raw, maxlag) # Consider only the lags which are smaller than maxlag
B_sf_full = np.copy(B_sf_raw[:, idx_valid])
# Lags & variance of the spectra
h_full, Vb_sf_full = experimental_frequency_variogram(h_raw_valid, B_sf_full)
# Smooth the raw FV
Vb_sf_full_smt = smoothed_frequency_variogram(h_full, Vb_sf_full, deg)


# Using FV class
cfv = FrequencyVariogramRaw(grid_spacing = dx_fine, maxlag = maxlag)
cfv.set_positions(dx_fine* p_full)
cfv.set_data(a_st, M)
cfv.compute_fv()
cfv.smooth_fv(deg)
fv_full = cfv.get_fv()
lags_full = cfv.get_lags()


print('#================================#')
print('lags == h_full? {}'.format(np.allclose(lags_full, h_full)))
print('fv_raw == Vb_sf_full_smt? {}'.format(np.allclose(fv_full, Vb_sf_full_smt)))
print('max(Vb_sf_full_smt) = {}'.format(Vb_sf_full_smt.max()))
print('max(fv_full) = {}'.format(fv_full.max()))
print('#================================#')

#%% Spatial sub-sampling

# Position selection, corresponds to the fine grid (= dx_fine)
p_smp = batch_subsampling(N_ss, N_scan, seed = seedNo_ss) # Sampling positions as indices, unitless

# Remove s0 from the sampled positions
idx_del = np.argwhere(np.logical_and(p_smp[:, 0] == p0_coarse[0, 0], p_smp[:, 1] == p0_coarse[0, 1]))
# Check if s0 is in the sampled locations
if len(idx_del) != 0:
    idx_del = idx_del[0, 0]
    p_smp = np.delete(p_smp, idx_del, axis = 0) # Remove s0 from the sampled positions
s_smp = p_smp* dx_fine
N_scan = len(s_smp)

scan_map = np.zeros((N_batch, N_batch))
scan_map[p_smp[:, 0], p_smp[:, 1]] = 1
scan_map[p0_fine[0, 0], p0_fine[0, 1]] = -1

# plot_cscan(scan_map.T, 
#             'Scan positions, coverage = ca. {}'.format(coverage), 
#             dx_coarse, dx_coarse)


# Measurement data of the selected sampling positions
formatter = SmartInspectDataFormatter(p_smp, 
                                      xmin = 0, xmax = N_batch, 
                                      ymin = 0, ymax = N_batch)
a_st_smp = formatter.get_data_matrix_pixelwise(a_st[:, p_smp[:, 0], p_smp[:, 1]]) # size = M x N_batch x N_batch

# plot_cscan(np.max(np.abs(a_st_smp), axis = 0).T, 
#             'Sampled data (C-Scan), coverage = ca. {}'.format(coverage), 
#             dx_coarse, dx_coarse)


#%% FV estimation of samples
# Space-freq. spectra of the sampled data
A_sf_smp = A_sf[:, p_smp[:, 0], p_smp[:, 1]] # size = len(f) x N_scan

# Raw lags & space-freq. spectra of the incremental process
d_raw_smp, B_sf_smp = incremental_process(p_smp, A_sf_smp)
h_raw_smp = dx_fine* d_raw_smp # [m]
# Limit the spatial lags to maxlag
idx_valid_smp, h_raw_smp_valid = limit_lag(h_raw_smp, maxlag) # Consider only the lags which are smaller than maxlag

# Lags & variance of the spectra
h_smp, Vb_sf_smp = experimental_frequency_variogram(h_raw_smp_valid, B_sf_smp[:, idx_valid_smp])

# FV using the FV class
cfv.set_positions(dx_fine* p_smp)
cfv.set_data(a_st_smp, M)
cfv.compute_fv()
cfv.smooth_fv(deg, lags_fv = lags_full)
fv_smp = cfv.get_fv()
lags_smp = cfv.get_lags()



#%% Block Frequency Kriging
""" Kriging in freq. domain
Goal = predicting the freq. spectrum of the A-Scan at an unscanned positions s0 from the neighboring A-Scans 

    A_hat(s0, f) =  np.dot(np.dot(g0hat(s0).T, F^{-1}), B_sf_smp)
"""

# Choose which FV to be used: computed only from samples vs using entire A-Scans in the batch
if use_smp == True:
    print('FV: Using samples!')
    h = np.copy(h_smp)
    Vb_sf = np.copy(Vb_sf_smp)
# Use smoothed FV
elif smooth == True:
    print('FV: using all A-Scans and FV is smoothed!')
    h = np.copy(h_full)
    Vb_sf_full_smt = smoothed_frequency_variogram(h_full, Vb_sf_full, deg)
    Vb_sf = np.copy(Vb_sf_full_smt)
# Use the raw experimental FV of the entire batch
else:
    print('FV: Using all A-Scans!')
    h = np.copy(h_full)
    Vb_sf = np.copy(Vb_sf_full)


# Limit the samples to the ones within the maxlag range
d0 = scdis.cdist(p_smp, p0_fine).flatten('F') # Spatial disance b/w s0 & the sampled
h0 = dx_fine* d0
idx_valid, h0 = limit_lag(h0, maxlag) # Limit the lag to the maxlag
p_within = p_smp[idx_valid, :] # Scan positions within the maxlag as indices, unitless
rank = len(np.unique(h0)) # Rank of F = unique combination of pair-wise distance
# Scan map, relevant to prediction, i.e. within the maxlag
ngb_map = np.zeros(scan_map.shape)
ngb_map[p_within[:, 0], p_within[:, 1]] = 1
ngb_map[p0_fine[0, 0], p0_fine[0, 1]] = -1
plot_cscan(ngb_map.T, 'Sampled positions < maxlag ', dx_fine, dx_fine)

A_sf_smp = A_sf[:, p_within[:, 0], p_within[:, 1]] # size = len(f) x N_scan

# Expected semivariances b/w s0 & the sampled positions in space-freq. domain
g0hat_sf = np.repeat(Vb_sf[:, -1].reshape(Vb_sf.shape[0], 1), len(h0), axis = 1)
for col, curr_h in enumerate(h0):
    g0hat_sf[:, col] = Vb_sf[:, np.argwhere(h == curr_h)].flatten('F')
    
    
# # Lags b/w scanned positions and corresponding indces -> same for all freq. bins
dict_idx = {}
for k, p_k in enumerate(p_within):
    # Lags b/w s_k & the other sampled positions
    d_k = scdis.cdist(p_within, np.array([p_k])) 
    h_k = dx_fine* d_k #[m]
    idx_valid, _ = limit_lag(h_k, maxlag) # Limit the lag to the maxlag
    
    # Indices of Vb_sf which correspond to h_k 
    indices = np.array([list(map(lambda x: np.where(x == h)[-1], h_k[idx_valid]))]) # Indices of h which corresponds to h_k
    
    # Save the indices
    dict_idx.update({
        str(k): {
            'samples' : idx_valid, 
            'FV': indices.flatten('F')
            }
        })      


#### Frequency Kriging ####
print('Freq. Kriging!')
print('#=======================================#')
start = time.time()

A0hat_sf = 1j*np.zeros(len(freq))
W = np.zeros((len(f_interest), len(h0)))

for f_bin in f_interest:
    # Variance matrix F of A_sf_smp for the current frequency
    # i-th column vec. of F = space-freq. covariances b/w the i-th sampled position & the rest 
    F = (Vb_sf[f_bin, :].max())* np.ones((len(h0), len(h0)))
    
    # Iterate over each sampled position
    for k in range(p_within.shape[0]):
        idx_smps = dict_idx[str(k)]['samples']
        idx_fv = dict_idx[str(k)]['FV']
        F[idx_smps, k] = Vb_sf[f_bin, idx_fv] # Pick values from Vb_sk according to h_k
        
        
    # Check F is real, symmetric positive definite matrix
    check_positive_definiteness(F)
    
    # Reduce the rank of F
    if p_within.shape[0] > rank:
        F = denoise_svd(F, rank)
    
    w = kriging_weights_sf_domain(F, g0hat_sf[f_bin, :], reg_Tik = reg_Tik, alpha = alpha_fac*np.sqrt(Pn))
    
    W[f_bin - fmin, :] = np.copy(w)
    # Prediction 
    A0hat_sf[f_bin] = np.dot(w, A_sf_smp[f_bin, :])
    
    
print('End of FK')
display_time(round(time.time() - start, 3))

a0hat = vmax* fft.irfft(A0hat_sf, n = M)

#a0hat = np.concatenate((np.zeros(Nt_offset), a0hat_roi))

# Show the l2 norm of the error
err_fk_noisy = a_st_noisy[:, p0_fine[0, 0], p0_fine[0, 1]] - a0hat
err_fk_clean = a_st_clean[:, p0_fine[0, 0], p0_fine[0, 1]] - a0hat

print('FK error(l2): vs noisy sig = {}'.format(np.round(np.linalg.norm(err_fk_noisy), 5)))
print('FK error(l2): vs clean sig = {}'.format(np.round(np.linalg.norm(err_fk_clean), 5)))


#%% Inverse-Distance Weighting (IDW)
a_smp = a_st[:, p_within[:, 0], p_within[:, 1]]
w_idw = 1/h0
w_idw = w_idw/np.sum(w_idw)

a0hat_idw = vmax* np.dot(a_smp, w_idw)

#%% Plots
#plt.close('all')

plt.figure()
plt.plot(a_st_clean[:, p0_fine[0, 0], p0_fine[0, 1]], label = 'true (clean)')
plt.plot(a0hat_idw, label = 'IDW')
plt.legend()

plt.figure()
plt.plot(a_st_clean[:, p0_fine[0, 0], p0_fine[0, 1]] - a0hat_idw, label = 'err_idw')
plt.plot(a_st_clean[:, p0_fine[0, 0], p0_fine[0, 1]] - a0hat, label = 'err_FK')
plt.legend()
plt.title('Error (vs clean signal)')

plt.figure()
plt.plot(a_st_clean[:, p0_fine[0, 0], p0_fine[0, 1]], label = 'true (clean)')
plt.plot(a0hat, label = 'FK')
plt.legend()


plt.figure()
plt.title('Clean vs noisy')
plt.plot(a_st_clean[:, p0_fine[0, 0], p0_fine[0, 1]], label = 'true (clean)')
plt.plot(a_st_noisy[:, p0_fine[0, 0], p0_fine[0, 1]], label = 'true (noisy)')
plt.legend()


plt.figure()
plt.plot(np.abs(A_sf[:, p0_fine[0, 0], p0_fine[0, 1]]), label = 'true')
plt.plot(np.abs(A0hat_sf), label = 'FK')
plt.xlabel('freq.')
plt.ylabel('Amplitude')
plt.title('A0_sf')
plt.legend()


# plt.figure()
# plt.plot(h_full, Vb_sf_full[15:51, :].T)
# plt.title('Vb_sf_full vs lag for each freq. bin [15, 50]')
# plt.xlabel('lag')

# plt.figure()
# plt.plot(h_smp, Vb_sf_smp[15:50, :].T)
# plt.title('Vb_sf_smp vs lag for each freq. bin [15, 50]')
# plt.xlabel('lag')


# # Plots
# from ultrasonic_imaging_python.visualization.slice_figures import SliceFigure3D

# z_axis = np.flip(np.arange(spc_param['Ntdata']))#* dz* 10**3 #[mm]
# x_axis = np.arange(spc_param['Nxdata'])#* dx* 10**3 #[mm]
# y_axis = np.arange(spc_param['Nydata'])#* dx* 10**3 #[mm]
# z_label = 'z'
# x_label = 'x'
# y_label = 'y'

# fig_y = SliceFigure3D(a_st, 
#                       2, 'Synthesized data along y-axis', 
#                       axis_range = [z_axis, x_axis, y_axis], 
#                       axis_label = [z_label, x_label, y_label], 
#                       initial_slice = 0, display_text_info = True, info_height = 0.35
#                       )

# plt.show()
