#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAFT example code
"""
import numpy as np
import time
import pickle

from ultrasonic_imaging_python.forward_models.propagation_models_3D_single_medium import PropagationModel3DSingleMedium
from ultrasonic_imaging_python.reconstruction import saft_grided_algorithms
from ultrasonic_imaging_python.utils.progress_bars import TextProgressBarFast
from ultrasonic_imaging_python.definitions import units
ureg = units.ureg

from tools.display_time import display_time
from tools.npy_file_writer import save_data
from tools.npy_file_writer import num2string
from tools.datetime_formatter import DateTimeFormatter


start_all = time.time()

#%% Functions

def saft_and_fft(spc_param, dt, dz, ml_param, f_range, fdate):
    # Unroll the parameters
    Nx = spc_param['Nxdata']
    Ny = spc_param['Nydata']
    M = spc_param['Ntdata']
    # ML params
    depth = ml_param['depth'] # = Nt_offset
    Ndef = ml_param['Ndef']
    dataNo = ml_param['dataNo']
    
    # Load data
    path = 'npy_data/ML/training/clean_data/depth_{}/Ndef_{}'.format(depth, num2string(Ndef))
    fname = '{}.npy'.format(num2string(dataNo))
    data = np.load('{}/{}'.format(path, fname))
    
    # FWM setting -> depth (= Nt_offset)
    spc_param.update({
            'zImage': -depth * dz * ureg.meter,
            't0': depth* dt * ureg.second
            })
    fname_saft = 'saft_matrix/Nx{}Ny{}M{}_depth_{}_{}'.format(Nx, Ny, M, depth, fdate)
    
    # FWM initialization
    model = PropagationModel3DSingleMedium()
    model.set_parameters(spc_param)
    model.set_apodizationmodel('Bartlett', apd_param)
    model.set_pulsemodel('Gaussian', pulse_params)
    
    # SAFT 
    print('start SAFT')
    saft_engine = saft_grided_algorithms.SAFTEngine(model, matrix_type="NLevelBlockToeplitz", 
                                                    enable_file_IO = True, filename = fname_saft)
    print('Begin Reconstruction')
    progress_bar = TextProgressBarFast(10, 'Forward model progress', 1, 1)
    reco = saft_engine.get_reconstruction(data[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, :], 
                                              progress_bar=progress_bar)
    progress_bar.finalize()
    
    # FFT
    fresp = np.fft.rfft(reco, n = M, axis = 0)
    
    # Free up the memory
    del data, reco
    
    return fresp[f_range, :, :]

#%% FWM parameters
# Specimen
c0 = 5900 #[m/S]
# ROI
Nx = 30
Ny = 30
M = 512

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

apd_param = {# Apodization parameters
    'max_angle': 25 * ureg.degree
    }
pulse_params = {# Pulse parameters
    'pulseLength': l_pulse,
    'fCarrier': fC * ureg.hertz,
    'B': 0.3,
    'fS': fS * ureg.hertz
    }

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
        #zImage': -Nt_offset * dz * ureg.meter,
        'xImage': 0 * 0.5 * 10**-3 * ureg.meter,
        'yImage': 0 * 0.5 * 10**-3 * ureg.meter,
        #'t0': Nt_offset* dt * ureg.second
        }

#%% ML parameters

# Dataset relevant
""" For dataset w/ fdate = '210427'
Clean data parametes: -> 240 varieties
    * depth
    * Ndef
    * dataNo

Data modification parameters:
    * phi (= phase)
    * noise power / dt
    
Output (FV) parameters: -> 125 varieties
    * p_batch (= batch starting point)
    * f (= freq. bin)

Input (subsampled data) parameters: -> 20 varieties
    * sampling coverage (or positions)
    
ML parameter dictionary:
    keys = ['depth', 'Ndef', 'dataNo', 'p_batch', 'phi', 'f_bin', 'ss_method', 'ss_cov', 'Pndt']
"""
N_batch = 10
fdate = '210427'
n_data = 240 # Variations w.r.t. the clean data
n_output = 125 # Variations w.r.t. the output data (= FV)
n_input = 20 # Variations w.r.t. the input data (= spatial subsampling)
# Intervals for loading/reconstructing data 
itv_measdata = n_output* n_input 
# Ranges for FFT to save memory
f_range = np.arange(51) # freq. bin

# Load ML parameter dictionray
with open('params/ml_data_params_{}.pickle'.format(fdate), 'rb') as handle:
    ml_param_all = pickle.load(handle)
    
# For save the reco results
path = 'npy_data/ML/training/reco'
    
#%% Compute reco
# Iterate over all ML parameter sets
for setNo in range(len(ml_param_all)):
    print('#===========================================#')
    print('setNo = {} / {}'.format(setNo, len(ml_param_all)))
    start = time.time()
    # Current parmeter set
    ml_param = ml_param_all[str(setNo)]
    
    # Reco & FFT 
    if (setNo % itv_measdata) == 0:
        fresp_roi = saft_and_fft(spc_param, dt, dz, ml_param, f_range, fdate)
    
    # Choose the freq. response of the current batch & f_bin
    x, y = ml_param['p_batch']
    f_bin = ml_param['f_bin']
    curr_fresp = fresp_roi[f_bin, x : x + N_batch, y : y + N_batch].flatten('F')
    data2save = np.concatenate((np.array([curr_fresp.real]), np.array([curr_fresp.imag])), axis = 0).T # size = 100 x 2
    print('Current params:')
    print('depth = {}, Ndef = {}, dataNo = {}'.format(ml_param['depth'], ml_param['Ndef'], ml_param['dataNo']))
    print('p_batch = {}, f_bin = {}'.format(ml_param['p_batch'], f_bin))
    
    # Save
    fname = 'setNo{}.npy'.format(setNo)
    save_data(data2save, path, fname)
    
    display_time(round(time.time() - start, 3))
    

# For printing the current time
dtf = DateTimeFormatter()
now = dtf.get_time_str()

print('##########################################################')
print('End of reco')
display_time(round(time.time() - start_all, 3))
print('with current time = {}'.format(now))
print('##########################################################')



