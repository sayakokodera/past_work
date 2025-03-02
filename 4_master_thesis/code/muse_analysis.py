# -*- coding: utf-8 -*-
"""
MUSE reconstruction
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import scipy.signal as scsi
import scipy.fft as fft
import scipy.spatial.distance as scdis
import time

import skgstat as skg

from ultrasonic_imaging_python.visualization.slice_figures import SliceFigure3D

from fwm.tof_calculator import ToFforDictionary2D
from fwm.dictionary_former import DictionaryFormer
from tools.display_time import display_time

from smart_inspect_data_formatter import SmartInspectDataFormatter
from spatial_subsampling import batch_subsampling
from signal_denoising import denoise_svd

plt.close('all') 

#======================================================================================================= Functions ====#
def get_opening_angle(c0, D, fC):
    r""" The opening angle (= beam spread), theta [grad], of a trnsducer element can be calculated with
        np.sin(theta) = 1.2* c0 / (D* fC) with
            c0: speed of sound [m/S]
            D: element diameter [m]
            fC: career frequency [Hz]
    (Cf: https://www.nde-ed.org/EducationResources/CommunityCollege/Ultrasonics/EquipmentTrans/beamspread.htm#:~:text=Beam%20spread%20is%20a%20measure,is%20twice%20the%20beam%20divergence.)
    """
    theta_rad = np.arcsin(1.2* c0/ (D* fC))
    return np.rad2deg(theta_rad) 


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


def variogram_single_slice(coords, values, n_lags, maxlag = None, ret_bins = False):
    V = skg.Variogram(coords, values, n_lags = n_lags, maxlag = maxlag)
    var_exp = V.experimental # = experimental variogram
    bins = V.bins
    del V
    
    if ret_bins == False:
        return var_exp
    else:
        return bins, var_exp
    
    
def variogram_single_slice_full(data_roi, N_batch, sliceNo_z, x_start, y_start, n_lags, ret_bins = False):
    # All grid points within the batch
    xx, yy = np.meshgrid(np.arange(x_start, x_start + N_batch), np.arange(y_start, y_start + N_batch))
    coords_x = xx.flatten()
    coords_y = yy.flatten()
    coords = np.array([coords_x, coords_y]).T
    # Select the corresponding measurement data
    values = data_roi[sliceNo_z, coords_x, coords_y]
    #values = values / np.abs(values).max() # Normalize -> not looks good, why?
    
    # Distance bins & variogram
    bins, var_exp = variogram_single_slice(coords, values, n_lags, ret_bins = True)
    
    if ret_bins == False:
        return var_exp
    else:
        return bins, var_exp

    
def spatiotemporal_variogram_sparse(data_roi, coords, n_lags, ret_bins = False):
    coords_x, coords_y = coords[:, 0], coords[:, 1]
    
    Var_exp = np.zeros((n_lags, M)) # Base of the variogram
    for sliceNo_z in range(M):
        values = data_roi[sliceNo_z, coords_x, coords_y]
        Var_exp[:, sliceNo_z] = variogram_single_slice(coords, values, n_lags)
        
    if ret_bins == False:
        return Var_exp
    else:
        bins, _ = variogram_single_slice(coords, values, n_lags, ret_bins = True)
        return bins, Var_exp
    
    
def spatiotemporal_variogram_full(starting_point):
    """ A function to be used with np.apply_along_axis
    
    Parameter
    ---------
        starting_point : np.array([x_start, y_start])
            Starting point to define the batch
    """
    x_start, y_start = starting_point # Starting point for the current batch
    print('x_start = {}, y_start = {}'.format(x_start, y_start))
    
    Var_exp = np.zeros((n_lags, M)) # Base of the variogram
    for sliceNo_z in range(M):
        Var_exp[:, sliceNo_z] = variogram_single_slice_full(data_roi, N_batch, sliceNo_z, x_start, y_start, n_lags)
        
    return Var_exp


def pairwise_difference(val):
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
    
    


def incremental_process(p, A_sf):
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
    (= experimental frequency variogram(FV)?)
    
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
    
    """
    Nf = B_sf.shape[0]
    h = np.concatenate((np.zeros(1), np.unique(h_raw)))
    
    # Experimental FV = expected value (over spatial lag) of space-freq. periodogram of the incremental process 
    Vb_sf = np.zeros((Nf, len(h)))
    # Exp. FV calculation for each (fixed) spatial lag
    for col, curr_dist in enumerate(h):
        if col == 0:
            pass
        else:
            idx = np.flatnonzero(h_raw == curr_dist) # Indices corresponding to the current lag
            Rbb = B_sf[:, idx].conj()* B_sf[:, idx]
            # Check if Rbb is real and positive
            if len(np.where(np.isreal(Rbb).astype(int) == 0)[0]) != 0:        
                raise ValueError('Rbb is NOT real with lag = {}'.format(curr_dist))
                #print('Rbb is real with lag = {}'.format(curr_dist))
            elif len(np.where(Rbb < 0)[0]) != 0:
                raise ValueError('Rbb is NOT positive with lag = {}'.format(curr_dist))
            
            Vb_sf[:, col] = np.mean(Rbb, axis = 1) # Mean over the lag
            
    # Check if Vb_sf is real and positive semi-definite
    if len(np.where(np.isreal(Vb_sf).astype(int) == 0)[0]) != 0:        
        raise ValueError('Experimental FV is NOT real')
    elif len(np.where(Vb_sf < 0)[0]) != 0:
        raise ValueError('Experimental FV is NOT positive semidefinite!')
    
    return h, Vb_sf



def plot_2D_Gaussian(hist, valmax, title, xlabel, ylabel):
    # Axis setting
    xx, yy = np.meshgrid(np.arange(hist.shape[0]), np.linspace(-valmax, valmax, Hist.shape[1]))
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
del data

plot_cscan(np.max(np.abs(data_roi), axis = 0).T, 'Data in our ROI (C-Scan)', dx, dx)

#%% Parameters
# Parameters for spatial subsampling
N_batch = 10
n_lags = 10 
maxlag = 0.5* np.sqrt(2)* N_batch
coverage = 0.1
N_scan = int(coverage* N_batch**2)
seedNo = 442#np.random.randint(600) #105

# Starting point for the batch
x_start = 20#max(0, np.random.randint(Nx - N_batch)) 
y_start = 20#max(0, np.random.randint(Ny - N_batch))

# Parameters for FV
s0 = np.array([[5, 5]]) # Scan position to predict its A-Scan, sould be in an array form for scdis.cdist
rank = 1# For low-rank approx. of Vb_sf
use_smp = False # True, if FV to be calculated using only the samples

#%% Reduce data size into a batch
data_batch = data_roi[:, x_start:(x_start + N_batch), y_start: (y_start + N_batch)]
#data_batch = data[zmin:zmax, x_start:(x_start + N_batch), y_start: (y_start + N_batch)]

plot_cscan(np.max(np.abs(data_batch), axis = 0).T, 
            'Actual batch data (C-Scan) @ x_start = {}, y_start = {}'.format(x_start, y_start), 
            dx, dx)

#%% Freq. analysis
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
    
"""
#Space-time domaint
a_st = data_batch #- np.mean(data_batch) # exclude the DC offset

# Space-freq. domain
freq = fS* 10**-6* fft.fftfreq(M) # = [0, f_Ny], f_Ny is included!
A_sf = fft.fft(a_st, axis = 0) # in space-freq. domain, i.e. preserving the data format
A_f = np.sum(A_sf, axis = (1, 2)) # spectra in freq. domain

# Angular-freq. domain
#A_af = fft.fft(A_sf, axis = 1)
A_af = fft.fft(fft.fft(A_sf, axis = 1), axis = 2) # FFT once or twice?
mean_A_af = np.mean(A_af) # which is NOT zero-mean!!!!!


# pair-wise distance: using scipy.spatial.distance.pdist??
"""
To-Do
(1) Check the noise power: get it from A_sf
(2) Get B_sf (which should be with the size = 100C2 = 50*99)
(3) Get the variance of B_sf
(4) Estimate the polynomial  
"""
# Noise power
Pn = np.mean(np.mean((np.abs(A_sf[100:, :, :])**2), axis = 0))

# Limit the freq. range
fmin, fmax = 0, len(freq)#15, 35
f_interest = np.arange(fmin, fmax)

# Incremental process (b_st, B_sf)
# All scan positions
xx, yy = np.meshgrid(np.arange(x_start, x_start + N_batch), np.arange(y_start, y_start + N_batch)) # Scan grids
pos = np.array([xx.flatten(), yy.flatten()]).T #  Scans are taken @ all grid points

# Raw lags & space-freq. spectra of the incremental process
h_raw, B_sf_raw = incremental_process(pos, np.reshape(A_sf, (len(freq), N_batch**2), 'F'))

# Limit the spatial lags to maxlag
idx_valid, h_raw_valid = limit_lag(h_raw, maxlag) # Consider only the lags which are smaller than maxlag
# Low-rank approx. of B_sf_full -> suppress the "exaggeration" of the signal -> why????
if rank == len(freq):
    print('Full rank!')
    B_sf_full = np.copy(B_sf_raw) # Ful rank [:, idx_valid]
else:
    B_sf_full = denoise_svd(B_sf_raw, rank) # Low-rank approx. [:, idx_valid]

# Lags & variance of the spectra
h_full, Vb_sf_full = experimental_frequency_variogram(h_raw_valid, B_sf_full)


# Check the validity of the complex-Gaussian assumption
Nbins = 30
valmax = max(B_sf_full.real.max(), B_sf_full.imag.max())

Hist = np.zeros((len(f_interest), Nbins, 2)) # 1st slice = real part, 2nd slice = imaginary part
for idx, f_bin in enumerate(f_interest):
    f_bin = fmin + idx
    Hist[idx, :, 0], _ = np.histogram(B_sf_full[idx, :].real, Nbins, range = (-valmax, valmax))
    Hist[idx, :, 1], _ = np.histogram(B_sf_full[idx, :].imag, Nbins, range = (-valmax, valmax))
# Remove the DC elements from the imaginary part 
Hist[0, :, 1] = np.nan

# Plots
#plot_2D_Gaussian(Hist[:, :, 0], valmax, 'Histogram: real part of B_sf', 'Freq. bins', 'Values')
#plot_2D_Gaussian(Hist[:, :, 1], valmax, 'Histogram: imaginary part of B_sf', 'Freq. bins', 'Values')


#%% Check the temporal second-order stationarity
""" For ST-Kriging, it is assumed that the process is intrinsic stationary in both space & time domain, meaning that 
the incremental process for a certain scan position and a fixed spatial lag is zero-mean WSS (Cf. Rao_17_IntrinsicFV).

The incremental process for a scan position s0 and a fixed lag h:
    b0_h[t, k] = a_st[t, sk] - a_st[t, s0] 
        with sk = s0 + h
Then, the intrinsic-stationarity is 
    * Mean(b0_h) = 0 
    * Var(b0_st[:, h]) = Variogram[h, t = 0]
"""

# Calculate the raw incremental process
data_mat = np.reshape(data_batch, (M, N_batch**2), 'F')
a0 = data_batch[:, s0[0, 0], s0[0, 1]]
b0_st = data_mat - a0[:, np.newaxis] # Raw incremental process in space-time domain

# Possible spatial lags
h0_full_raw = scdis.cdist(pos, s0).flatten('F')
h0_full = np.unique(h0_full_raw) # h_full

# Compute the histogram of the b0_st for each lag
val_max = np.abs(b0_st).max()
val_min = -val_max
d_val = 0.05
bins = np.arange(val_min, val_max + d_val, d_val)
Hist_b0 = np.zeros((len(bins[:-1]), len(h0_full)))
for col, lag in enumerate(h0_full):
    indices = np.flatnonzero(h0_full_raw == lag) # Indices corresponding to the current lag
    Hist_b0[:, col], edges = np.histogram(b0_st[:, indices], bins = bins)


#%% Spatial sub-sampling
# Position selection
p_smp = batch_subsampling(7, N_scan, seed = seedNo) # Sampling positions
# Remove s0 from the sampled positions
idx_del = np.argwhere(np.logical_and(p_smp[:, 0] == s0[0, 0], p_smp[:, 1] == s0[0, 1]))
# Check if s0 is in the sampled locations
if len(idx_del) != 0:
    idx_del = idx_del[0, 0]
    p_smp = np.delete(p_smp, idx_del, axis = 0) # Remove s0 from the sampled positions
N_scan = len(p_smp)

scan_map = np.zeros((N_batch, N_batch))
scan_map[p_smp[:, 0], p_smp[:, 1]] = 1

plot_cscan(scan_map.T, 
            'Scan positions @ x_start = {}, y_start = {}, coverage = ca. {}'.format(x_start, y_start, coverage), 
            dx, dx)


# Measurement data of the selected sampling positions
formatter = SmartInspectDataFormatter(p_smp, 
                                      xmin = 0, xmax = N_batch, 
                                      ymin = 0, ymax = N_batch)
data_sampled = formatter.get_data_matrix_pixelwise(data_batch[:, p_smp[:, 0], p_smp[:, 1]]) # size = M x N_batch x N_batch

plot_cscan(np.max(np.abs(data_sampled), axis = 0).T, 
            'Sampled data (C-Scan) @ x_start = {}, y_start = {}, coverage = ca. {}'.format(x_start, y_start, coverage), 
            dx, dx)


# Space-freq. spectra of the sampled data
A_sf_smp = A_sf[:, p_smp[:, 0], p_smp[:, 1]] # size = len(f) x N_scan

# Raw lags & space-freq. spectra of the incremental process
h_raw_smp, B_sf_smp = incremental_process(p_smp, A_sf_smp)
# Limit the spatial lags to maxlag
_, h_raw_smp_valid = limit_lag(h_raw_smp, maxlag) # Consider only the lags which are smaller than maxlag

#Low-rank approx. of B_sf_full -> suppress the "exaggeration" of the signal -> why????
if rank == len(freq):
    print('Full rank!')
    B_sf_smp_apr = np.copy(B_sf_smp) # Ful rank
else:
    B_sf_smp_apr = denoise_svd(B_sf_smp, rank) # Low-rank approx.
    
# Lags & variance of the spectra
h_smp, Vb_sf_smp = experimental_frequency_variogram(h_raw_smp_valid, B_sf_smp_apr)


#%% Experimental variogram in space-teim domain
# bins, var_exp = spatiotemporal_variogram_sparse(data_roi, coords, n_lags, ret_bins = True)

# plot_spatiotemporal_variogram(var_exp, 
#                               'Variogram @ x_start = {}, y_start = {}, coverage = ca. {}'.format(
#                                       x_start, y_start, coverage
#                               ), 
#                               M, bins)#* dx* 10**3)

#%% Frequency Kriging
""" Kriging in freq. domain
Goal = predicting the freq. spectrum of the A-Scan at an unscanned positions s0 from the neighboring A-Scans 

    A_hat(s0, f) =  np.dot(np.dot(g0hat(s0).T, F^{-1}), B_sf_smp)
"""

# Choose which FV to be used: computed only from samples vs using entire A-Scans in the batch
if use_smp == True:
    print('FV: Using samples!')
    h = np.copy(h_smp)
    Vb_sf = np.copy(Vb_sf_smp)
else:
    print('FV: Using all A-Scans!')
    h = np.copy(h_full)
    Vb_sf = np.copy(Vb_sf_full)


# Limit the samples to the ones within the maxlag range
h0 = scdis.cdist(p_smp, s0).flatten('F') # Spatial disance b/w s0 & the sampled 
idx_valid, h0 = limit_lag(h0, maxlag) # Limit the lag to the maxlag
s_within = p_smp[idx_valid, :] 
ngb_map = np.zeros(scan_map.shape)
ngb_map[s_within[:, 0], s_within[:, 1]] = 1
plot_cscan(ngb_map.T,'Sampled positions < maxlag ', dx, dx)

A_sf_smp = A_sf[:, s_within[:, 0], s_within[:, 1]] # size = len(f) x N_scan

# Expected semivariances b/w s0 & the sampled positions in space-freq. domain
g0hat_sf = np.repeat(Vb_sf[:, -1].reshape(Vb_sf.shape[0], 1), len(h0), axis = 1)
for col, curr_h in enumerate(h0):
    g0hat_sf[:, col] = Vb_sf[:, np.argwhere(h == curr_h)].flatten('F')


# Frequency Kriging
print('Freq. Kriging!')
print('#=======================================#')
start = time.time()

A0hat_sf = 1j+ np.zeros(len(freq))
W = np.zeros((len(f_interest), len(h0)))

for f_bin in f_interest:
    # Variance matrix F of A_sf_smp for the current frequency
    # i-th column vec. of F = space-freq. "semi-variances" b/w the i-th sampled position & the rest 
    F = Vb_sf[f_bin, -1]* np.ones((len(h0), len(h0)))
    # Iterate over each sampled position
    for k, s_k in enumerate(s_within):
        # Lags b/w s_k & the other sampled positions
        h_k = scdis.cdist(s_within, np.array([s_k])) 
        idx_valid, _ = limit_lag(h_k, maxlag) # Limit the lag to the maxlag
        # Assign the elements of Vb_sf according to h_k 
        indices = np.array([list(map(lambda x: np.where(x == h)[-1], h_k[idx_valid]))]) # Indices of h which corresponds to h_k
        F[idx_valid, k] = Vb_sf[f_bin, indices.flatten('F')] # Pick values from Vb_sk according to h_k
    
    # Kriging weights for the current freq. bin
    Finv = np.linalg.inv(F)
    w = np.dot(g0hat_sf[f_bin, :].reshape(1, len(h0)), Finv) # Weights
    W[f_bin - fmin, :] = np.copy(w)
    # Prediction 
    A0hat_sf[f_bin] = np.dot(w, A_sf_smp[f_bin, :])
    
    
print('End of FK')
display_time(round(time.time() - start, 3))

a0hat = fft.ifft(A0hat_sf, M) 


plt.figure()
plt.plot(data_batch[:, s0[0, 0], s0[0, 1]], label = 'true')
#plt.plot(fft.irfft(A_sf[:, 5, 5], M), label = 'IRFFT')
plt.plot(a0hat, label = 'FK')
plt.legend()


#%% Inverse-Distance Weighting
a_smp = a_st[:, s_within[:, 0], s_within[:, 1]]
w_id = 1/h0
w_id = w_id/np.sum(w_id)

a0hat_id = np.dot(a_smp, w_id)

plt.figure()
plt.plot(data_batch[:, s0[0, 0], s0[0, 1]], label = 'true')
plt.plot(a0hat_id, label = 'Inverse-Distance')
plt.legend()

plt.figure()
plt.plot(data_batch[:, s0[0, 0], s0[0, 1]] - a0hat_id, label = 'err_id')
plt.plot(data_batch[:, s0[0, 0], s0[0, 1]] - a0hat, label = 'err_FK')
plt.legend()
plt.title('Error')

#%% Temporal variogram for fully covered data
# z_start = 495#95
# var_T = np.zeros((10, N_batch**2))

# for y in range(N_batch):
#     for x in range(N_batch):
#         col = y* N_batch + x
#         val = data_roi[z_start + np.arange(N_batch), x_start + x, y_start + y]
        
#         # Check if all values are same -> skip interpolation (otherwise skg.Variogram fails)
#         if len(set(val)) < 2:
#             pass
#         else:
#             V = skg.Variogram(np.array([np.arange(N_batch), np.zeros(N_batch)]).T, val, 
#                               maxlag = None)
#             var_T[:, col] = V.experimental # = experimental variogram

# plot_spatiotemporal_variogram(var_T, 
#                               'Temporal variogram @ x_start = {}, y_start = {}, coverage = ca. {}'.format(
#                                       x_start, y_start, coverage
#                               ), 
#                               N_batch**2, V.bins)
                              
# data_voxel = data_batch[z_start:z_start + N_batch, :, :]
# d_full = data_voxel.flatten('F')


# #%% Plots: data
# z_axis = np.arange(M)#* dz* 10**3 #[mm]
# x_axis = np.arange(Nx)#* dx* 10**3 #[mm]
# y_axis = np.flip(np.arange(Ny))#* dx* 10**3 #[mm]
# z_label = 'z'
# x_label = 'x'
# y_label = 'y'

# # Sampled data in the selected batch
# fig_z = SliceFigure3D(data_sampled, 
#                       0, 'Sampled data slice (@ x_start = {}, y_start = {}) along z-axis'.format(x_start, y_start), 
#                       [z_axis, np.arange(N_batch), np.flip(np.arange(N_batch))], 
#                       [z_label, x_label, y_label], 
#                       z_start, False, display_text_info = True, info_height = 0.35)


# # Batch slice along z-axis
# fig_z = SliceFigure3D(data_batch, 
#                       0, 'Batch slice (@ x_start = {}, y_start = {}) along z-axis'.format(x_start, y_start), 
#                       [z_axis, np.arange(N_batch), np.flip(np.arange(N_batch))], 
#                       [z_label, x_label, y_label], 
#                       z_start, False, display_text_info = True, info_height = 0.35)

# # Slice along z-axis
# fig_z = SliceFigure3D(data_roi, 0, 'MUSE data slice along z-axis', 
#                       [z_axis, x_axis, y_axis], [z_label, x_label, y_label], 
#                       z_start, False, display_text_info = True, info_height = 0.35)

# # =============================================================================
# # # Slice along x-axis
# # fig_x = SliceFigure3D(data_roi, 1, 'MUSE data slice along x-axis', [z_axis, x_axis, y_axis], [z_label, x_label, y_label], 
# #                       0, False, display_text_info = True, info_height = 0.35)
# # 
# # # Slice along y-axis
# # fig_y = SliceFigure3D(data_roi, 2, 'MUSE data slice along y-axis', [z_axis, x_axis, y_axis], [z_label, x_label, y_label], 
# #                       0, False, display_text_info = True, info_height = 0.35)
# # =============================================================================

# plt.show()

