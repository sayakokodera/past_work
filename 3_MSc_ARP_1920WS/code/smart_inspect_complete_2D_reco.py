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
#import fastmat as fm
#import fastmat.algorithms as fma



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


def plot_reco(figure_no, data, title, dx, dz):
    fig = plt.figure(figure_no)
    ax = fig.add_subplot(111)
    ax.imshow(data)
    ax.set_aspect(dz/dx)
    ax.set_title(title)
    ax.set(xlabel = 'x / dx', ylabel = 'z / dz')


# SAFT based reco
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
    

# %% SI measurment data
date = '200903_1'
dtype = 'A_pixel'
path = 'npy_data/SmartInspect/{}'.format(date)
data_3D = np.load('{}/{}.npy'.format(path, dtype))

# ROI
y = 50
t_forerun = 0 #[S], offset time which is eliminated from the measurement data to remove the front wall echo


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
# ROI 
Nx = data_3D.shape[1]
Nt_forerun = int(t_forerun* fS)
Nt_offset = Nt_forerun
Nz = Nt_forerun + data_3D.shape[0]
M = Nz - Nt_offset
# Spacing
dx = (260/1891.5)*10**-3 #[m] (255/1583)
dy = (130/945)*10**-3 #[m] (129/806)
dz = 0.5* c0/(fS)


# %% Measurement data handling
# Extract A-Sacns within the ROI
data_2D = data_3D[:M, :, y] 
data_2D = data_2D / np.abs(data_2D).max() # normalize the data
# Grid points for reconstruction matrix within ROI
p_scan = np.zeros((Nx, 2))
p_scan[:, 0] = np.arange(Nx)* dx

plot_reco(1, data_2D, 'Raw data (y = {})'.format(y), dx, dz) 
del data_3D

# %% SAFT Reconstruction w/o BEC 
# Set dimensions and the rest of the parameters
fwm_param = {'Nx' : Nx, 'Nz' : Nz, 'Nt' : Nz, 'c0' : c0, 'fS' : fS, 'fC' : fC, 'alpha' : alpha, 'dx' : dx,
             'opening_angle' : opening_angle}

# Call the error correction class
start_bec = time.time()
bec = BlindErrorCorrection2D(fwm_param, Nt_offset, r)
# Reco_track
start_recotrack = time.time()
#R_track = calculate_Reco(bec, data_vec, p_scan, with_envelope = with_envelope) 
H = bec.dictionary_formation(p_scan, with_envelope = False)
R_saft = np.reshape(np.dot(H.T, data_2D.flatten('F')), (M, Nx), 'F')
# Plots
plot_reco(3, R_saft, 'SAFT Reco (y = {}), no error correction'.format(y), dx, dz) 
print('R_saft calculation:')
display_time(time.time() - start_recotrack)


# %% FISTA using fastmat
# =============================================================================
# Niteration_fista = 30
# max_R_saft = np.abs(R_saft).max()
# Lambda = 0.75* max_R_saft
# 
# # Reco w/o error correction
# print('FISTA! (w/o error correction)')
# start = time.time()
# # Convert the measurement dictionary into fastmat matrix format
# matH = fm.Matrix(H)
# # Reco
# fista = fma.FISTA(matH, numLambda = Lambda, numMaxSteps = Niteration_fista)
# R_fista = np.reshape(fista.process(data_2D.flatten('F')), (M, Nx), 'F')
# display_time(round(time.time() - start, 3))
# plot_reco(5, R_fista, 'Reco FISTA (fastmat)', dx, dz)
# =============================================================================


#%% Save data
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
