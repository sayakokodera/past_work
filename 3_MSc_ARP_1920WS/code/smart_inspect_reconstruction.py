# -*- coding: utf-8 -*-
"""
#===========================================#
        SmartInspect reconstruction
#===========================================#
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.signal as scsi
import scipy.sparse.linalg as sclin

from blind_error_correction import BlindErrorCorrection2D
from signal_denoising import denoise_svd
#from fista import FISTA
import fastmat as fm
import fastmat.algorithms as fma


r"""   
To-Do: 20.06.20(Sat) & 20.06.21(Sun)
    X BEC: adjust the code (envelope, apodization)
    X Reconstruct the MUSE data
        -> successful


To-Do: 20.06.23(Tue)
    * Measurement 
        - min. 20% coverage
        - calibration w/ backwall ehcho for parameter estimation
            -> another measurement required! 
            In the current data, the amplitude is clipped and cannot be used for parameter analysis
        - More B-Scan-ish measurement
            (Trace of the current data might be too "straight" and not hyperbolic enough?)

To-Do: 20.06.24(Wed)
    * Parameter estimation w/ measuremnt data
    * Data smoothing 
    * 2D reco (w/o position correction)
    
To-Do: 
    * Denoise with wavelet
    * Area averaging with circular symmetric Gaussian window
    * BEC: Adjust the dictionary formation for error correction, s.t. the dictionary can be partially modified
    * 3D reco extension
    * defect_map_handling: extend to multiple defects
    * test BEC w/ the "real-sized" defect -> does it work?
"""


plt.close('all')   
# %% Functions 
def display_time(stop):
    if stop > 60 and stop < 60**2:
        print('** Time to complete : {} min'.format(round(stop/60, 2)))
    elif stop >= 60**2:
        print('** Time to complete : {} h'.format(round(stop/(60**2), 2)))
    else:
        print('** Time to complete : {} s'.format(round(stop, 2)))
 
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


def select_xlim(data, threshold):
    data_norm = data/np.abs(data).max()
    peaks = np.max(np.abs(data_norm), axis = 0)
    col_toosmall = np.argwhere(peaks < threshold)
    return col_toosmall


def plot_reco(figure_no, data, title, dx, dz):
    fig = plt.figure(figure_no)
    ax = fig.add_subplot(111)
    ax.imshow(data)
    ax.set_aspect(dz/dx)
    ax.set_title(title)
    ax.set(xlabel = 'x / dx', ylabel = 'z / dz')
    

def peak_positions(data):
    arr = np.zeros(data.shape)
    peak = np.argmax(np.abs(data), axis = 0)
    for x in range(data.shape[1]):
        arr[peak[x], x] = 1
    return arr


# %% SAFT based reco
def calculate_Reco(bec, ascans, p, with_envelope = False, with_Jacobian = False, deltax = None):
    M = bec.M
    Nx = bec.Nx
    print('Start calculating the dictionary!')
    # Without spatial approximation (for H_true and H_track)
    if with_Jacobian == False: 
        H = bec.dictionary_formation(p, with_envelope = with_envelope)
    # With spatial approximation (for H_opt)
    else:
        H_base, J_base = bec.dictionary_formation(p, with_envelope = with_envelope, with_Jacobian = True)
        H = H_base + np.dot(np.kron(np.diag(deltax), np.identity(M)), J_base)
        del H_base, J_base
    print('End of dictionary calculation')
    reco = np.dot(H.T, ascans)
    Reco = np.reshape(reco, (M, Nx), 'F')
    del H
    return Reco
    

def tracking_error_correction(bec, A_roi, p_scan, x_threshold, Niteration, epsilon, e_max, blur, triangular = False):
    # !!!!! BEC only works with the non-zero column vectors !!!!! 
    # (1) Reduce the data size: non-zero vector selection
    col_eliminate = select_xlim(A_roi, x_threshold) # Find the columns < x_threshold
    col_nz = np.delete(np.arange(A_roi.shape[1]), col_eliminate) # Indices of non-zero column vectors
    A_nz = A_roi[:, col_nz] # = Non-zero column vectors (noisy)
    A_nz_svd = denoise_svd(A_nz, d = 2) 
    p_nz = np.delete(p_scan, col_eliminate, axis = 0) # Adjust teh positional info t the reduced size
    del A_roi, A_nz
    
    # (2) Estimate the defect position via hyperbola fit 
    bec.estimate_defect_positions(A_nz_svd, p_nz[:, 0])
    # Generate the defect map (adjusted to the reduced size (including only the non-zero vectors))
    if blur == False: # Point source
        print('Point source!')
        bec.defect_map_conversion(Nx, col_eliminate)
    else: # Blurred = representing the physical size
        print('Blurring!')
        bec.defect_map_conversion(col_eliminate = col_eliminate, blur = True, sigma_z = 2, sigma_x = 0.3) 
    
    # (3) Error correction
    x_opt_nz, deltax_opt_nz, _ = bec.error_correction(p_nz, A_nz_svd.flatten('F'), Niteration, epsilon, e_max, 
                                                      triangular)
    print('Size of x_opt_nz = {}'.format(x_opt_nz.shape))
    # (4) Adjust the corrected positional info to the ROI size (including zero-vectors)
    p_opt = np.copy(p_scan)
    p_opt[col_nz, 0] = np.copy(x_opt_nz)
    deltax_opt = np.zeros(p_scan.shape[0])
    deltax_opt[col_nz] = np.copy(deltax_opt_nz)
    
    return p_opt, deltax_opt


def tracking_error_correction_SAFT(bec, data_vec, p_scan, Niteration, epsilon, e_max, triangular = False):
    
    # Initialization
    p = np.copy(p_scan)
    deltax = np.zeros(p_scan.shape[0])
    
    # Iteration
    for n in range(Niteration):
        print('#===================================#')
        print('{}-th iteration!'.format(n))
        print('1st scan positions = {}'.format(p[0]))
        # (2) Estimate the defect position via FISTA
        bec.b_hat = calculate_Reco(bec, data_vec, p, with_Jacobian = True, deltax = deltax).flatten('F')
        
        # (3) Error correction
        x, deltax, _ = bec.error_correction(p, data_vec, 1, epsilon, e_max, triangular)
        print('Size of x = {}'.format(x.shape))
        
        # (4) Update the measurement positions
        p[:, 0] = np.copy(x)
    
    return p, deltax

# %% FISTA based reco

# Self-implemented FISTA   
def calculate_Reco_FISTA_si(bec, ascans, p, Lambda, Niteration, with_Jacobian = False, deltax = None, 
                             with_envelope = False, ret_matrix = False):
    M = bec.M
    Nx = bec.Nx
    
    # Dictionary calculation
    print('Start calculating the dictionary!')
    start = time.time()
    # Without spatial approximation (for H_true and H_track)
    if with_Jacobian == False: 
        H = bec.dictionary_formation(p, with_envelope = with_envelope)
    # With spatial approximation (for H_opt)
    else:
        H_base, J_base = bec.dictionary_formation(p, with_envelope = with_envelope, with_Jacobian = True)
        H = H_base + np.dot(np.kron(np.diag(deltax), np.identity(M)), J_base)
        del H_base, J_base
    display_time(round(time.time() - start, 3))
    print('End of dictionary calculation')
    
    # Largest singular value = stepsize for FISTA
    print('Largest singular value calculation!')
    start = time.time()
    L = sclin.svds(H, tol = 10, maxiter = 80, return_singular_vectors = False)[0]**2
    print('Sigma = {}'.format(np.sqrt(L)))
    display_time(round(time.time() - start, 3))
    
    # Reco
    fista = FISTA(Lambda, Niteration)
    fista.compute(H, ascans, L = L)
    reco = fista.get_solution()
    del H
    
    if ret_matrix == False:
        return reco
    else:
        return np.reshape(reco, (M, Nx), 'F')

# FISTA using fastmat
def calculate_Reco_FISTA(bec, ascans, p, Lambda, Niteration, with_Jacobian = False, deltax = None, 
                         with_envelope = False, ret_matrix = False):
    M = bec.M
    Nx = bec.Nx
    
    # Dictionary calculation
    print('Start calculating the dictionary!')
    print('Using the einvelope? {}'.format(with_envelope))
    start = time.time()
    # Without spatial approximation (for H_true and H_track)
    if with_Jacobian == False: 
        H = bec.dictionary_formation(p, with_envelope = with_envelope)
    # With spatial approximation (for H_opt)
    else:
        H_base, J_base = bec.dictionary_formation(p, with_envelope = with_envelope, with_Jacobian = True)
        H = H_base + np.dot(np.kron(np.diag(deltax), np.identity(M)), J_base)
        del H_base, J_base
    display_time(round(time.time() - start, 3))
    print('End of dictionary calculation')
    
    # Convert the measurement dictionary into fastmat matrix format
    matH = fm.Matrix(H)
    # Reco
    fista = fma.FISTA(matH, numLambda = Lambda, numMaxSteps = Niteration)
    reco = fista.process(ascans)
    del H, matH
    
    if ret_matrix == False:
        return reco
    else:
        return np.reshape(reco, (M, Nx), 'F')

def tracking_error_correction_FISTA(bec, data_vec, p_scan, Niteration, epsilon, e_max, Lambda, with_envelope = False,
                                    triangular = False):
    
    # Initialization
    p = np.copy(p_scan)
    deltax = np.zeros(p_scan.shape[0])
    
    # Iteration
    for n in range(Niteration):
        print('#===================================#')
        print('{}-th iteration!'.format(n))
        print('1st scan positions = {}'.format(p[0]))
        # (2) Estimate the defect position via FISTA
        bec.b_hat = calculate_Reco_FISTA(bec, data_vec, p, Lambda, 1, with_Jacobian = True, deltax = deltax,
                                         with_envelope = with_envelope) 
        
        # (3) Error correction
        x, deltax, _ = bec.error_correction(p, data_vec, 1, epsilon, e_max, triangular)
        print('Size of x = {}'.format(x.shape))
        
        # (4) Update the measurement positions
        p[:, 0] = np.copy(x)
    
    return p, deltax



# %% SI measurment data
date = '200708_2'
dtype = 'A_pixel'
path = 'npy_data/SmartInspect/{}'.format(date)
#A_meas = np.load('{}/A_aa_25pixels.npy'.format(path))
A_meas = np.load('{}/{}.npy'.format(path, dtype))

# %% Variables  
# ROI
xmin, xmax = 10, A_meas.shape[1] - 10
y = 42
t_forerun = 9.5*10**-6 #[S], offset time which is eliminated from the measurement data to remove the front wall echo
tmin, tmax = 230, 330#240, 340#
x_threshold = 0.1 # Threshold for choosing "non-zero" vectors for BEC hyp-fit
with_envelope = False

# Hyperbola fit
blur = False # if True, the estimated point scatterer is blurred to mimic a physical-sized defect

# Newton method variables
Niteration = 10
epsilon = 0.02
e_max_norm = 10

# %% Parameter Setting 
# Measurement setting
c0 = 5920 #[m/S]
fS = 80*10**6 #[Hz] 
D = 3.5* 10**-3 #[m], diameter of the transducer element
fC = 4.48*10**6 #[Hz] 
opening_angle = get_opening_angle(c0, D, fC) #[deg]
wavelength = c0/ fC # [m]
# Data model parameter (estimated from the measurement data)
alpha = 8.9*10**12#8.29*10**12 #[Hz]**2, bandwidth factor
phi = 270.5#271#277.22 #[grad], phase
r = -0.02#-0.0316 # unitless (-1...1), Chirp rate
# ROI & Data matrix info
Nx = xmax - xmin
Nt_forerun = int(t_forerun* fS)
Nt_offset = Nt_forerun + tmin
Nz = Nt_forerun + tmax
M = tmax - tmin
dx = (260/1560)*10**-3 #[m] (255/1583)
dy = (130/778)*10**-3 #[m] (129/806)
dz = 0.5* c0/(fS)

# %% Measurement data handling
# Extract A-Sacns within the ROI
A_roi = A_meas[:M, xmin:xmax, y] 
A_roi = A_roi / np.abs(A_roi).max() # normalize the data
# Grid points for reconstruction matrix within ROI
p_scan = np.zeros((Nx, 2))
p_scan[:, 0] = np.arange(Nx)* dx

#Env = np.abs(scsi.hilbert(A_roi, axis = 0))
#Peaks = peak_positions(Env)s

# Plots
plot_reco(1, A_roi, 'Raw data (y = {})'.format(y), dx, dz) 
#plot_reco(2, Peaks, 'Peak position of raw data (y = {})'.format(y), dx, dz) 
#del A_meas

# %% SAFT Reconstruction w/o BEC 
# Set dimensions and the rest of the parameters
fwm_param = {'Nx' : Nx, 'Nz' : Nz, 'Nt' : Nz, 'c0' : c0, 'fS' : fS, 'fC' : fC, 'alpha' : alpha, 'dx' : dx,
             'opening_angle' : opening_angle}

# Call the error correction class
start_bec = time.time()
bec = BlindErrorCorrection2D(fwm_param, Nt_offset, r)

# Setting up the data vector for the error correction
# data_vec is used only for the error correction (for reconstruction, using A_roi.flatten('F'))
A_roi_svd = denoise_svd(A_roi, d = 2)# Denoising
if with_envelope == False:
    data_vec = A_roi_svd.flatten('F')
else:
    Env = np.abs(scsi.hilbert(A_roi_svd, axis = 0))
    data_vec = Env.flatten('F')
del A_roi_svd

# Reco_track
start_recotrack = time.time()
#R_track = calculate_Reco(bec, data_vec, p_scan, with_envelope = with_envelope) 
H = bec.dictionary_formation(p_scan, with_envelope = with_envelope)
R_track = np.reshape(np.dot(H.T, data_vec), (M, Nx), 'F')
# Plots
plot_reco(3, R_track, 'SAFT Reco (y = {}), no error correction'.format(y), dx, dz) 
print('R_track calculation:')
display_time(time.time() - start_recotrack)

#del R_track

# %% FISTA reconstruction
# =============================================================================
# Niteration_fista = 50
# max_Rtrack = 223.1
# Lambda = 0.7* max_Rtrack
# 
# # Without error correction
# R_fista = calculate_Reco_FISTA(bec, data_vec, p_scan, Lambda, Niteration_fista, ret_matrix = True)
# 
# plot_reco(4, R_fista, 'FISTA Reco (y = {}), no error correction'.format(y), dx, dz) 
# del R_fista
# 
# # With error correction
# p_opt, deltax_opt = tracking_error_correction_FISTA(bec, A_roi, p_scan, Niteration_fista, epsilon, e_max_norm* dx, 
#                                                     Lambda)
# 
# R_fista_opt = calculate_Reco_FISTA(bec, data_vec, p_opt, Lambda, Niteration_fista, with_Jacobian = True, 
#                                    deltax = deltax_opt, ret_matrix = True)
# 
# plot_reco(5, R_fista_opt, 'FISTA Reco (y = {}), with error correction'.format(y), dx, dz) 
# =============================================================================

# %% FISTA using fastmat

# =============================================================================
# #H = np.load('H_200708_2_y42.npy')
# R_track = np.load('R_track.npy')
# plot_reco(3, R_track, 'SAFT Reco (y = {}), no error correction'.format(y), dx, dz) 
# 
# Niteration_fista = 30
# max_Rtrack = np.abs(R_track).max()
# Lambda = 0.75* max_Rtrack
# 
# del R_track
# 
# # Reco w/o error correction
# print('FISTA! (w/o error correction)')
# start = time.time()
# R_fista = calculate_Reco_FISTA(bec, A_roi.flatten('F'), p_scan, Lambda, Niteration_fista, ret_matrix = True)
# display_time(round(time.time() - start, 3))
# plot_reco(5, R_fista, 'Reco FISTA (fastmat)', dx, dz)
# 
# # Error correction
# print('Error correction using FISTA!')
# start = time.time()
# p_opt, deltax_opt = tracking_error_correction_FISTA(bec, data_vec, p_scan, Niteration_fista, epsilon, e_max_norm* dx, 
#                                                     Lambda, with_envelope = with_envelope)
# print('#============================#')
# print('End of the error correction')
# display_time(round(time.time() - start, 3))
# 
# # Reco
# R_fista_opt = calculate_Reco_FISTA(bec, A_roi.flatten('F'), p_opt, Lambda, Niteration_fista, with_Jacobian = True, 
#                                    deltax = deltax_opt, ret_matrix = True)
# plot_reco(6, R_fista_opt, 'FISTA Reco (y = {}), with error correction'.format(y), dx, dz) 
# 
# =============================================================================
# %% SAFT Reconstruction w/ BEC 
# =============================================================================
# # Reco_opt
# start_recoopt = time.time()
# 
# ### Error correction
# p_opt, deltax_opt = tracking_error_correction(bec, A_roi, p_scan, x_threshold, Niteration, epsilon, 
#                                               e_max_norm* dx, blur = blur)
# 
# #p_opt, deltax_opt = tracking_error_correction_SAFT(bec, data_vec, p_scan, 50, epsilon, e_max_norm* dx)
# 
# ### Reco
# R_opt = calculate_Reco(bec, A_roi.flatten('F'), p_opt, False, True, deltax_opt)
# # Plots
# plot_reco(4, R_opt, 'Reco (y = {}), with error correction'.format(y), dx, dz)
# print('Reco_true calculation:')
# display_time(time.time() - start_recoopt)
# #del R_opt
# 
# ### Estimated defect map: adjusted to the ROI size (including zero columnvectors)
# bec.defect_map_conversion(blur = blur, sigma_z = 5, sigma_x = 5)
# B_hat = np.reshape(bec.b_hat, (A_roi.shape), 'F')
# if blur == False:
#     plot_reco(8, B_hat, 'Estimated def map, point source', dx, dz)
# else:
#     plot_reco(8, B_hat, 'Estimated def map, blurred', dx, dz)
# 
# # Positions
# plt.figure(7)
# plt.plot(p_scan[:, 0]*10**3, 'r--', label = 'Tracked positions')
# plt.plot((p_opt[:, 0] + deltax_opt)*10**3, 'b', label = '"Optimized"')
# plt.title('Error correction')
# plt.xlabel('x/dx')
# plt.ylabel('Positions [mm]')
# plt.legend()
# =============================================================================


#Save data
from tools.npy_file_writer import save_data
# =============================================================================
# save_data(A_roi_noisy, 'npy_data/SmartInspect/{}/y{}_{}'.format(date, y, dtype), 'A_roi_noisy.npy')
# save_data(R_track, 'npy_data/SmartInspect/{}/y{}_{}'.format(date, y, dtype), 'R_track.npy')
# save_data(R_opt, 'npy_data/SmartInspect/{}/y{}_{}'.format(date, y, dtype), 'R_opt.npy')
# save_data(B_hat, 'npy_data/SmartInspect/{}/y{}_{}'.format(date, y, dtype), 'B_hat.npy')
# save_data(np.hstack((np.array([p_track[:, 0]]).T, np.array([p_opt[:, 0]]).T)), 
#           'npy_data/SmartInspect/{}/y{}_{}'.format(date, y, dtype), 'ptrack_vs_popt.npy')
# save_data(Env, 'npy_data/SmartInspect/{}/y{}_{}'.format(date, y, dtype), 'Env_noisy.npy')
# 
# =============================================================================
