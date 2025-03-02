# -*- coding: utf-8 -*-
"""
#=====================================#
        Multi-Scatterers Analysis
#=====================================#
"""
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as nlin
import scipy.signal as scsi
import time

from defect_map_handling import DefectMapSingleDefect2D
from gaussian_blur import gaussian_blur 
from opening_angle_calculator import get_opening_angle
from blind_error_correction import BlindErrorCorrection2D
from signal_denoising import denoise_svd
from display_time import display_time


plt.close('all')   
# %% Functions 
def plot_reco(figure_no, data, title, dx, dz):
    fig = plt.figure(figure_no)
    ax = fig.add_subplot(111)
    ax.imshow(data)
    ax.set_aspect(dz/dx)
    ax.set_title(title)
    ax.set(xlabel = 'x / dx', ylabel = 'z / dz')
    
# Frequency domain analysis
def freq_anaysis(ascan):
    # Analytic signal (= Gaussian window)
    analytic = scsi.hilbert(ascan)
    # Spectrum
    sp = np.fft.fft(analytic, 500)
    # Dominant freq. component
    idx = np.argmax(np.abs(sp))
    # Phase of the dominant component
    phi = np.arctan(sp[idx].imag/sp[idx].real) #[rad]
    return np.abs(sp), phi


# %% Parameter Setting 
# Measurement setting: mimic the real SI measurement 200706
c0 = 5920 #[m/S]
fS = 80*10**6 #[Hz] 
D = 3.5* 10**-3 #[m], diameter of the transducer element
fC = 4.48*10**6 #[Hz] 
opening_angle = get_opening_angle(c0, D, fC) #[deg]
wavelength = c0/ fC # [m]
# Data model parameter (estimated from the measurement data)
alpha = 8.9*10**12#8.29*10**12 #[Hz]**2, bandwidth factor
phi = 271#277.22 #[degree], phase for "modulation"
r = -0.02#-0.0316 # unitless (-1...1), Chirp rate
# ROI & Data matrix info
Nx = 50 # -> variable
tmin, tmax = 240, 340
Nz = tmax
Nt_offset = tmin
M = tmax - tmin
dx = (260/1382)*10**-3 #[m] (260/1382)
dy = (130/696)*10**-3 #[m] (129/806)
dz = 0.5* c0/(fS)
# FWM parameters for dictionary formation
fwm_param = {'Nx' : Nx, 'Nz' : Nz, 'Nt' : Nz, 'c0' : c0, 'fS' : fS, 'fC' : fC, 'alpha' : alpha, 'dx' : dx,
             'opening_angle' : opening_angle}

# %% Variable (1): scan positions
p_scan = np.zeros((Nx, 2)) # = [[x1, z1], [x2, z2] ....] in [m]
p_scan[:, 0] =  np.arange(Nx)* dx 

# %% Variables (2): defect positions
# 1. defect = center of the ROI
p_def1 = np.array([(Nx/2)* dx, (Nz - 0.5*M)* dz]) # = [x_def, z_def] in [m]
# 2. defect = right next to the 1. defect
p_def2 = np.array([(Nx/2 + 1)* dx, (Nz - 0.5*M)* dz]) # = [x_def, z_def] in [m]

# Convert to the defectmap: 1. defect
dm2d = DefectMapSingleDefect2D(p_def1, Nx, Nz, dx, dz)
dm2d.generate_defect_map_multidim(Nt_offset)
b_ps1 = dm2d.get_defect_map_1D()
B_ps1 = dm2d.get_defect_map_2D()
plot_reco(1, B_ps1, 'Defect map 1 (point source)', dx, dz)

# =============================================================================
# # Convert to the defectmap: 2. defect
# dm2d = DefectMapSingleDefect2D(p_def2, Nx, Nz, dx, dz)
# dm2d.generate_defect_map_multidim(Nt_offset)
# b_ps2 = dm2d.get_defect_map_1D()
# B_ps2 = dm2d.get_defect_map_2D()
# plot_reco(2, B_ps2, 'Defect map 2 (point source)', dx, dz)
# 
# # 3. defect = point source excatly between ps1 and ps2
# b_ps3 = 0.5* (b_ps1 + b_ps2)
# B_ps3 = np.reshape(b_ps3, (M, Nx), 'F')
# plot_reco(3, B_ps3, 'Defect map 3 (point source)', dx, dz)
# =============================================================================

# Blurred defect map
#B_bl = gaussian_blur(M, Nx, [int(z_def_idx - Nt_offset), x_def_idx], 5, 5)
#b_bl = B_bl.flatten('F')
#plot_reco(2, B_bl, 'Defect map ("physical" sized scatterer)', dx, dz)

# %% FWM: synthetic data
# Dictionary formation
print('Dictionary formation!')
start = time.time()
bec = BlindErrorCorrection2D(fwm_param, Nt_offset, r)
H = bec.dictionary_formation(p_scan) 
display_time(np.around(time.time() - start, 3))

# Synthetic data
A_1 = np.reshape(np.dot(H, b_ps1), (M, Nx), 'F') 
#A_2 = np.reshape(np.dot(H, b_ps2), (M, Nx), 'F') 
#A_3 = np.reshape(np.dot(H, b_ps3), (M, Nx), 'F') 

#del H

plot_reco(4, A_1, 'B-Scan w/ dx = {}mm'.format(round(dx*10**3, 3)), dx, dz)
#plot_reco(5, A_2, 'B-Scan w/ 2. defect', dx, dz)
#plot_reco(6, A_3, 'B-Scan w/ 3. defect', dx, dz)

# %% Reconstruction with FISTA
#from fista import FISTA
import scipy.sparse.linalg as sclin

# Parameters
Lambda = 0.8
Niteration = 1

# =============================================================================
# # Increase the spatial sampling distance
# rate = 2
# Nx_fista = rate* Nx
# bec.Nx = Nx_fista
# p_fista = np.zeros((Nx_fista, 2))
# p_fista[:, 0] = dx* np.arange(Nx_fista)
# A_fista = np.zeros((M, Nx_fista))
# A_fista[:, :Nx] = np.copy(A_1)
# 
# # Dictionary
# H_fista = bec.dictionary_formation(p_fista) 
# # Largest singular value
# =============================================================================
print('SVD!')
start = time.time()
L = sclin.svds(H, tol = 10, maxiter = 50, return_singular_vectors = False)[0]**2
display_time(round(time.time() - start, 3))

# =============================================================================
# print('#==========================#')
# print('FISTA!')
# start = time.time()
# 
# fista = FISTA(Lambda, Niteration)
# fista.compute(H, A_1.flatten('F'), L = L)
# r = fista.get_solution()
# 
# display_time(np.around(time.time() - start, 3))
# plot_reco(5, np.reshape(r, (M, bec.Nx), 'F'), 'Reco w/ dx = {}mm'.format(round(dx*10**3, 3)), dx, dz)
# =============================================================================


# %% FISTA w/ fastmat
import fastmat as fm
import fastmat.algorithms as fma
#from ultrasonic_imaging_python.math_tools.matrices import FastmatWrapper

# Convert the measurement dictionary into fastmat matrix format
print('FISTA!')
start = time.time()

matH = fm.Matrix(H)
fista = fma.FISTA(matH, numLambda=0.8, numMaxSteps=50)
r = fista.process(A_1.flatten('F'))

display_time(round(time.time() - start, 3))

plot_reco(5, np.reshape(r, (M, bec.Nx), 'F'), 'Reco w/ dx = {}mm'.format(round(dx*10**3, 3)), dx, dz)






