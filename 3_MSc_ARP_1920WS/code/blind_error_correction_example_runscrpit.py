# -*- coding: utf-8 -*-
"""
BEC exmple runscript 
"""
import numpy as np
import matplotlib.pyplot as plt

from blind_error_correction import BlindErrorCorrection2D

#=============================================================================================== Parameter Setting ====#
# Fixed parameters
opening_angle = 28 #[deg]
c0 = 6300 #[m/S]
fS = 80*10**6 #[Hz] 
fC = 5*10**6 #[Hz] 
alpha = 20*10**12 #[Hz]**2
dx = 0.5*10**-3 #[m]
dz = 0.5* c0/(fS) 
wavelength = 1.26* 10**-3 # [m] 
 
# Size of the specimen
size_x = 20* 10**-3 #[m]
size_z = 70* 10**-3 #[m]
# Actual dimension of the speciment
Nx = size_x / dx 
Nz = size_z / dz 

# Defect position
x_def_mm = size_x/2 #[m]
x_def_idx = int(x_def_mm/dx)
z_def_mm = 20* 10**-3 #[m]
z_def_idx = int(z_def_mm/dz)
p_def_idx = np.array([x_def_idx, z_def_idx])
# ROI dimensions
Nz_roi = 200
Nt_offset = Nz - Nz_roi
Nx_roi = int(2* (z_def_mm* np.tan(np.deg2rad(0.5*opening_angle)))/dx)

# FWM parameters
fwm_param = {'Nx' : Nx, 'Nz' : Nz, 'Nt' : Nz, 'c0' : c0, 'fS' : fS, 'fC' : fC, 'alpha' : alpha, 'dx' : dx}


# Tracking error
err_norm = 1# Bound for the tracking error, normalized with the wavelength
# Error distribution
triangular = False # if False, use uniform distribution
# Newton method variables
Niteration = 10
epsilon = 0.02


