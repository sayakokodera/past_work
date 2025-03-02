#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAFT example code
"""
import numpy as np
import matplotlib.pyplot as plt


from ultrasonic_imaging_python.forward_models.propagation_models_3D_single_medium import PropagationModel3DSingleMedium
from ultrasonic_imaging_python.reconstruction import saft_grided_algorithms
from ultrasonic_imaging_python.utils.progress_bars import TextProgressBarFast
from ultrasonic_imaging_python.definitions import units
ureg = units.ureg

from phase_shifter import PhaseShifter
from tools.datetime_formatter import DateTimeFormatter

#%% Parameters
# Specimen
c0 = 5900 #[m/S]

# ROI
Nx = 10
Ny = 10
M = 512
Nt_offset = 647 #391, 647, 903
Nt_full = Nt_offset + M

# Grid spacing
dx = 0.5* 10**-3 #[m]
dy = 0.5* 10**-3 #[m]

# Measurement parameters
fS = 80* 10**6 #[Hz]
dt = 1/fS #[S]
dz = 0.5* c0* dt #[m]

# Pulse setting
l_pulse = 128
fC = 3.36*10**6 #[Hz]

# Defect setting
Ndefect = 7
seedNo_def = 267
amp_low = 0.4


apd_param = {# Apodization parameters
    'max_angle': 25 * ureg.degree
    }
pulse_params = {# Pulse parameters
    'pulseLength': l_pulse,
    'fCarrier': fC * ureg.hertz,
    'B': 0.3,
    'fS': fS * ureg.hertz
    }

# SAFT matrix file setting
save = False
dtf = DateTimeFormatter()
today = dtf.get_date_str()
fname_saft = 'saft_matrix/Nx{}Ny{}M{}_depth_{}'.format(Nx, Ny, M, Nt_offset)


#%% Load data
from tools.npy_file_writer import num2string

# file settings
Ndefect = 10
dataNo = 87
path = 'npy_data/ML/train/clean_data/depth_{}/Ndef_{}'.format(Nt_offset, num2string(Ndefect))
fname = '{}.npy'.format(num2string(dataNo))
phi = 180 # [degree]
# Load
data_roi_phi0 = np.load('{}/{}'.format(path, fname))[:, :Nx, :Ny]

# Shift phase
ps = PhaseShifter()
data_roi = ps.shift_phase(data_roi_phi0, phi, axis = 0)


#%% SAFT: all at once

# FWM setting
spc_param = {# Specimen parameters
    'dxdata': dx * ureg.meter,
    'dydata': dy * ureg.meter,
    'c0': c0 * ureg.meter / ureg.second,
    'fS': fS * ureg.hertz,
    'Nxdata': Nx,
    'Nydata': Ny,
    'Ntdata': M,
    'Nxreco': Nx,
    'Nyreco': Ny,
    'Nzreco': M,
    'anglex': 0 * ureg.degree,
    'angley': 0 * ureg.degree,
    'zImage': -Nt_offset * dz * ureg.meter,
    'xImage': 0 * 0.5 * 10**-3 * ureg.meter,
    'yImage': 0 * 0.5 * 10**-3 * ureg.meter,
    't0': Nt_offset* dt * ureg.second
    }

# FWM initialization
model = PropagationModel3DSingleMedium()
model.set_parameters(spc_param)
model.set_apodizationmodel('Bartlett', apd_param)
model.set_pulsemodel('Gaussian', pulse_params)


# Reco
print('start SAFT')
saft_engine = saft_grided_algorithms.SAFTEngine(model, matrix_type="NLevelBlockToeplitz", 
                                                enable_file_IO = save, filename = fname_saft)
print('Begin Reconstruction')
progress_bar = TextProgressBarFast(10, 'Forward model progress', 1, 1)
reco_roi = saft_engine.get_reconstruction(data_roi[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, :], 
                                          progress_bar=progress_bar)
progress_bar.finalize()

#%% SAFT: batch wise

# =============================================================================
# # Choose the batch
# N_batch = 10
# x_start, y_start = 0, 15
# data1 = data_roi[:, x_start: x_start + N_batch, y_start : y_start + N_batch]
# data2 = data_roi[:, x_start + N_batch : x_start + 2*N_batch, y_start : y_start + N_batch]
# data3 = data_roi[:, x_start + 2*N_batch : x_start + 3*N_batch, y_start : y_start + N_batch]
# 
# # FWM setting
# spc_param = {# Specimen parameters
#     'dxdata': dx * ureg.meter,
#     'dydata': dy * ureg.meter,
#     'c0': c0 * ureg.meter / ureg.second,
#     'fS': fS * ureg.hertz,
#     'Nxdata': N_batch,
#     'Nydata': N_batch,
#     'Ntdata': M,
#     'Nxreco': N_batch,
#     'Nyreco': N_batch,
#     'Nzreco': M,
#     'anglex': 0 * ureg.degree,
#     'angley': 0 * ureg.degree,
#     'zImage': -Nt_offset * dz * ureg.meter,
#     'xImage': 0 * 0.5 * 10**-3 * ureg.meter,
#     'yImage': 0 * 0.5 * 10**-3 * ureg.meter,
#     't0': Nt_offset* dt * ureg.second
#     }
# 
# # FWM initialization
# model = PropagationModel3DSingleMedium()
# model.set_parameters(spc_param)
# model.set_apodizationmodel('Bartlett', apd_param)
# model.set_pulsemodel('Gaussian', pulse_params)
# 
# 
# # Reco
# print('start SAFT')
# saft_engine = saft_grided_algorithms.SAFTEngine(model, matrix_type="NLevelBlockToeplitz", 
#                                                 enable_file_IO = save, filename = fname)
# print('Begin Reconstruction')
# progress_bar = TextProgressBarFast(10, 'Forward model progress', 1, 1)
# reco1 = saft_engine.get_reconstruction(data1[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, :], 
#                                       progress_bar=progress_bar)
# progress_bar.finalize()
# 
# print('Begin Reconstruction')
# progress_bar = TextProgressBarFast(10, 'Forward model progress', 1, 1)
# reco2 = saft_engine.get_reconstruction(data2[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, :], 
#                                       progress_bar=progress_bar)
# progress_bar.finalize()
# 
# print('Begin Reconstruction')
# progress_bar = TextProgressBarFast(10, 'Forward model progress', 1, 1)
# reco3 = saft_engine.get_reconstruction(data3[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, :], 
#                                       progress_bar=progress_bar)
# progress_bar.finalize()
# 
# reco_all = np.concatenate((reco1, reco2, reco3), axis = 1)
# #reco_all = reco_all / np.abs(reco_all).max()
# 
# =============================================================================

#%% Plots
from ultrasonic_imaging_python.visualization.slice_figures import SliceFigure3D

z_axis = np.flip(np.arange(spc_param['Ntdata']))#* dz* 10**3 #[mm]
x_axis = np.arange(spc_param['Nxdata'])#* dx* 10**3 #[mm]
y_axis = np.arange(spc_param['Nydata'])#* dx* 10**3 #[mm]
z_label = 'z'
x_label = 'x'
y_label = 'y'

# =============================================================================
# fig_y = SliceFigure3D(reco1, 
#                       2, 'SAFT reco1 along y-axis', 
#                       axis_range = [z_axis, x_axis, y_axis], 
#                       axis_label = [z_label, x_label, y_label], 
#                       initial_slice = 0, display_text_info = True, info_height = 0.35
#                       )
# 
# =============================================================================
fig_y = SliceFigure3D(data_roi, 
                      2, 'Synthesized data along y-axis', 
                      axis_range = [z_axis, np.arange(Nx), np.arange(Ny)], 
                      axis_label = [z_label, x_label, y_label], 
                      initial_slice = 0, display_text_info = True, info_height = 0.35
                      )

fig_y = SliceFigure3D(reco_roi, 
                      2, 'SAFT reco along y-axis', 
                      axis_range = [z_axis, np.arange(Nx), np.arange(Ny)], 
                      axis_label = [z_label, x_label, y_label], 
                      initial_slice = 0, display_text_info = True, info_height = 0.35
                      )

plt.show()




