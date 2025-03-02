# -*- coding: utf-8 -*-
"""
FWM example code
"""
import numpy as np
import matplotlib.pyplot as plt


from ultrasonic_imaging_python.forward_models.propagation_models_3D_single_medium import PropagationModel3DSingleMedium
from ultrasonic_imaging_python.forward_models.data_synthesizers import DataSynthesizer
from ultrasonic_imaging_python.reconstruction import saft_grided_algorithms

from ultrasonic_imaging_python.definitions import units
ureg = units.ureg

from defects_synthesizer import DefectsSynthesizer


#%% Parameters
# Specimen
c0 = 5900 #[m/S]

# ROI
Nx = 30
Ny = 30
M = 512
Nt_offset = 903 #391, 647, 903
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
save = True
fname = 'saft_matrix/Nx{}Ny{}M{}_depth_{}'.format(Nx, Ny, M, Nt_offset)

#%% Data generation: full version
### Full version
# =============================================================================
# spc_param = {# Specimen parameters
#     'dxdata': dx * ureg.meter,
#     'dydata': dy * ureg.meter,
#     'c0': c0 * ureg.meter / ureg.second,
#     'fS': fS * ureg.hertz,
#     'Nxdata': Nx,
#     'Nydata': Ny,
#     'Ntdata': Nt_full,
#     'Nxreco': Nx,
#     'Nyreco': Ny,
#     'Nzreco': Nt_full,
#     'anglex': 0 * ureg.degree,
#     'angley': 0 * ureg.degree,
#     'zImage': -0 * dz * ureg.meter,
#     'xImage': 0 * 0.5 * 10**-3 * ureg.meter,
#     'yImage': 0 * 0.5 * 10**-3 * ureg.meter,
#     't0': 0* dt * ureg.second
#     }
# # Defect positions: full
# defect_dict = {
#     'x': np.arange(Nx),
#     'y': np.arange(Ny),
#     'z': np.arange(int(Nt_offset + 0.5* l_pulse), int(Nt_offset +  M - 0.5* l_pulse)),
#     'amp_low' : amp_low
#     }
# ds = DefectsSynthesizer(Nt_full, Nx, Ny, defect_dict)
# ds.set_defects(Ndefect, seed_p = seedNo_def)
# defmap = ds.get_defect_map_3d() # shape = Nt_full x Nx x Ny
# 
# # FWM initialization
# model = PropagationModel3DSingleMedium()
# model.set_parameters(spc_param)
# model.set_apodizationmodel('Bartlett', apd_param)
# model.set_pulsemodel('Gaussian', pulse_params)
# 
# # Data generation
# synth = DataSynthesizer(model)
# d_full_raw = synth.get_data(defmap) # shape = Nt_full x 1 x 1 x 1 x 1 x Nxdata x Nydata
# d_full_raw = np.reshape(d_full_raw.flatten('F'), defmap.shape, 'F') # shape = Nt_full x Nxdata x Nydata
# # Adjust to teh ROI: trim the part 0...Nt_offset
# d_full = d_full_raw[Nt_offset:, :, :] # shape = M x Nx x Ny
# =============================================================================


#%% Data generation ROI version
# ROI version
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

# Defect positions: ROI
defect_dict = {
    'x': np.arange(Nx),
    'y': np.arange(Ny),
    'z': np.arange(int(0.5* l_pulse), int(M - 0.5* l_pulse)),
    'amp_low' : amp_low
    }
ds = DefectsSynthesizer(M, Nx, Ny, defect_dict)
ds.set_defects(Ndefect, seed_p = seedNo_def)
defmap = ds.get_defect_map_3d() # shape = M x Nx x Ny


# FWM initialization
model = PropagationModel3DSingleMedium()
model.set_parameters(spc_param)
model.set_apodizationmodel('Bartlett', apd_param)
model.set_pulsemodel('Gaussian', pulse_params)

# Data generation
synth = DataSynthesizer(model, enable_file_IO = save, filename = fname)
d_roi = synth.get_data(defmap) # shape = M x 1 x 1 x 1 x 1 x Nxdata x Nydata
d_roi = np.reshape(d_roi.flatten('F'), defmap.shape, 'F') # shape = M x Nxdata x Nydata





