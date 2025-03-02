# -*- coding: utf-8 -*-
"""
Example runsctript for synthetic data generation and reconstruction
"""
import numpy as np
import matplotlib.pyplot as plt
import time

from blind_error_correction import BlindErrorCorrection2D
from tools.npy_file_writer import save_data
from tools.datetime_formatter import DateTimeFormatter 
from image_quality_analyzer import ImageQualityAnalyzerSE
from visualization_2D_runscript import savepng


plt.close('all')   
#======================================================================================================= Functions ====#
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

def calculate_dimensions(z_def_mm, dz, opening_angle, dx, Nzimg):
    z_idx = int((z_def_mm* 10**-3)/dz)
    Nz = z_idx + int(Nzimg/2)
    Nx = int(2* (Nz*dz*np.tan(np.deg2rad(0.5*opening_angle)))/dx) #- 10
    if Nx < 21:
        Nx = 21
    elif Nx > 50:
        Nx = Nx - 10
    x_idx = int(Nx/2)
    return Nx, Nz, x_idx, z_idx


def calculate_ROI(Nt, Nzimg):
    Nt_offset = max(0, Nt - Nzimg)
    M = Nt - Nt_offset
    return Nt_offset, M


def calculate_Reco(bec, ascans, p, with_Jacobian = False, deltax = None):
    M = bec.M
    Nx = bec.Nx
    # Without spatial approximation (for H_true and H_track)
    if with_Jacobian == False: 
        H = bec.dictionary_formation(p)
    # With spatial approximation (for H_opt)
    else:
        H_base, J_base = bec.dictionary_formation(p, with_Jacobian = True)
        H = H_base + np.dot(np.kron(np.diag(deltax), np.identity(M)), J_base)
        del H_base, J_base
    reco = np.dot(H.T, ascans)
    Reco = np.reshape(reco, (M, Nx), 'F')
    del H
    return Reco
    
        
def plot_reco(figure_no, data, title, dx, dz):
    fig = plt.figure(figure_no)
    ax = fig.add_subplot(111)
    ax.imshow(data)
    ax.set_aspect(dz/dx)
    ax.set_title(title)
    ax.set(xlabel = 'x / dx', ylabel = 'z / dz')
    

def position_manipulation(e_max, p_true, triangular = False, p_scan_idx = None):
    np.random.seed(0)
    if triangular == False:
        err = np.random.uniform(-e_max, e_max, (len(p_true)))
    else:
        raise AttributeError('position_manipulation: currently works onlywith the uniform distribution!')
    
    if p_scan_idx is None:
        p_track = np.copy(p_true)
        p_track[:, 0] = p_track[:, 0] + err 
    else:
        p_track = np.zeros((len(p_scan_idx), 2))
        for i, p_idx in enumerate(p_scan_idx):
            p_track[i, 0] = p_true[p_idx, 0] + err[p_idx]
    return p_track


def tracking_error_correction(bec, A_track, p_track, Niteration, epsilon, e_max, triangular):
    # Remove zero vectors
    A_track_nz = A_track[:, np.any(A_track != 0, axis = 0)] # Raw data contains only non-zero vectors
    a_track_nz = A_track_nz.flatten('F')
    # Hyperbola fit -> estimate the defect position
    bec.hyperbola_fit(A_track_nz, p_track[:, 0])
    # Error correction
    x_opt, deltax_opt, _ = bec.error_correction(p_track, a_track_nz, Niteration, epsilon, e_max, triangular)
    # x_opt -> p_opt
    p_opt = np.zeros(p_track.shape)
    p_opt[:, 0] = x_opt
    return p_opt, deltax_opt


def data_allocation(A_true_base, p, dx, shape):
    # Round the positions to the nearest grid points
    p_idx = np.around(p[:, 0]/dx).astype(int)
    # Base of the A_scans
    A_hat = np.zeros(shape)
    for i, x in enumerate(p_idx):
        A_hat[:, x] = A_true_base[:, i]
    return A_hat

#=============================================================================================== Parameter Setting ====#
c0 = 6300 #[m/S]
fS = 80*10**6 #[Hz] 
D = 3.5* 10**-3 #[m], diameter of the transducer element
fC = 4*10**6 #[Hz] 
alpha = 20*10**12 #[Hz]**2
dx = 0.6*10**-3 #[m]
opening_angle = get_opening_angle(c0, D, fC) #[deg]
dz = 0.5* c0/(fS) 
wavelength = c0/ fC # [m]  
Nzimg = 200 

#======================================================================================================= Variables ====#
# Main variables
z_def_mm = 30# [mm]
e_norm = 0.5 # [lambda] Bound for the tracking error, normaliyed with the wavelength
Nscan_ratio = 0.2
p_seed = 8 # For scan position selection

# Error distribution
triangular = False # if False, use uniform distribution
# Newton method variables
Niteration = 10
epsilon = 0.02

#======================================================================================================================#
# Set dimensions and the rest of the parameters
Nx, Nz, x_idx, z_idx = calculate_dimensions(z_def_mm, dz, opening_angle, dx, Nzimg)
Nt_offset, M = calculate_ROI(Nz, Nzimg)
p_def_idx = np.array([x_idx, z_idx])
fwm_param = {'Nx' : Nx, 'Nz' : Nz, 'Nt' : Nz, 'c0' : c0, 'fS' : fS, 'fC' : fC, 'alpha' : alpha, 'dx' : dx,
             'opening_angle' : opening_angle}

# Scan positions = only partially scanned
Nscan = np.ceil(Nscan_ratio* Nx).astype(int)
np.random.seed(p_seed)
p_scan_idx =  np.sort(np.random.choice(Nx, size = Nscan))

# Call the error correction class
start_bec = time.time()
bec = BlindErrorCorrection2D(fwm_param, Nt_offset)

# Generate measurement data 
bec.defectmap_generation(p_def_idx)
bec.data_generation(p_scan_idx)
_, A_true = bec.get_data(True)
A_true_base = A_true[:, p_scan_idx] # Only the A-Scans recorded at the scan positions, zero vectors are eliminated
print('Data generation:')
display_time(time.time() - start_bec)

# Reco_true
start_recotrue = time.time()
p_true = bec.p_true
Reco_true = calculate_Reco(bec, bec.a_true, p_true) 
plot_reco(1, Reco_true, 'Reco_true', dx, dz) 
del Reco_true
print('Reco_true calculation:')
display_time(time.time() - start_recotrue)

# Position manipulation
e_max = e_norm* wavelength
p_track = position_manipulation(e_max, p_true, triangular = False, p_scan_idx = p_scan_idx)
A_track = data_allocation(A_true_base, p_track, dx, A_true.shape)

plt.figure(2)
plt.imshow(A_track)

# Reco_track -> works fine....?
start_recotrack = time.time()
Reco_track = calculate_Reco(bec, A_track.flatten('F'), p_true)
plot_reco(2, Reco_track, 'Reco_track', dx, dz)
del Reco_track
print('Reco_track calculation:')
display_time(time.time() - start_recotrack)

# Reco_opt -> currently no odification for the dictionary, artefacts are less intense (20.06.14)
start_recoopt = time.time()
p_opt, deltax_opt = tracking_error_correction(bec, A_track, p_track, Niteration, epsilon, e_max, triangular)
A_opt = data_allocation(A_true_base, p_opt, dx, A_true.shape)
Reco_opt = calculate_Reco(bec, A_opt.flatten('F'), p_true)
plot_reco(3, Reco_opt, 'Reco_opt', dx, dz)   
del Reco_opt
print('Reco_true calculation:')
display_time(time.time() - start_recoopt)
# =============================================================================
# Reco_opt = calculate_Reco(bec, p_opt, with_Jacobian = True, deltax = deltax_opt)
# plot_reco(3, Reco_opt, 'Reco_opt', dx, dz)   
# del Reco_opt
# print('Reco_true calculation:')
# display_time(time.time() - start_recoopt)
# =============================================================================


