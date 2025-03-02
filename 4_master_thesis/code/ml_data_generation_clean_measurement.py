# -*- coding: utf-8 -*-
"""
Data generation: clean measurement
"""

import numpy as np
import matplotlib.pyplot as plt
import time


from ultrasonic_imaging_python.forward_models.propagation_models_3D_single_medium import PropagationModel3DSingleMedium
from ultrasonic_imaging_python.forward_models.data_synthesizers import DataSynthesizer

from ultrasonic_imaging_python.definitions import units
ureg = units.ureg

from defects_synthesizer import DefectsSynthesizer
from tools.display_time import display_time
from tools.npy_file_writer import save_data
from tools.npy_file_writer import num2string


#%% Functions

def find_element_in_nested_list(mylist, elm):
    """ Find an element in a nested list, where no element is repeated
    """
    for sub_list in mylist:
        if elm in sub_list:
            return (mylist.index(sub_list), sub_list.index(elm))
    # If nothing is returned after checking all sub_lists
    raise ValueError("'{elm}' is not in list".format(elm = elm))


def generate_defect_map(Ndefect, dict_p_def, defect_dict, counter = None):
    # Parameters
    Nx = len(defect_dict['x'])
    Ny = len(defect_dict['y'])
    M = defect_dict['M']
     
    if Ndefect == 1:
        raise AttributeError('generate_defect_map: Ndefect == 1 is not currently supported!')
    
    else:
        ds = DefectsSynthesizer(M, Nx, Ny, defect_dict)
        ds.set_defects(Ndefect)
        defmap = ds.get_defect_map_3d() # shape = Nt_full x Nx x Ny
        return defmap
        
def file_export_setting(Nt_offset, Ndefect, counter):
    """ e.g. 
        npy_data/ML/training/clean_data/depth_0/Ndef_001/001.npy
    """
    # String setting
    str_Ndef = num2string(Ndefect)
    str_counter = num2string(counter)
    
    path = 'npy_data/ML/training/clean_data/depth_{}/Ndef_{}'.format(Nt_offset, str_Ndef)
    fname = '{}.npy'.format(str_counter)
    
    return path, fname


#%% Parameters: constant
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

#%% Parameters: variables

### Significant parameters: Nt_offset, Ndefect, p_def ###
# Nt_offset
Nt_min = int(5*10**-3 / dz)
range_Nt_offset = Nt_min + int(0.5* M)* np.arange(0, 5)
# Ndefect
Ndefect_min = 1 # included
Ndefect_max = 31 # excluded
range_Ndefect = np.array([2, 5, 10])#np.arange(Ndefect_min, Ndefect_max)
# p_def
ds_factor = 3 # for Ndefect == 1, where scan positions are systematically chosen
dict_p_def = { # number of datasets: varies with Ndefect -> bounds
        'bounds' : [[1], list(np.arange(2, 5)), list(np.arange(5, 11)), list(np.arange(11, Ndefect_max))],
        '0' : int(len(np.triu_indices(Nx)[0]) / ds_factor), # case : Ndefect == 1
        '1' : 200, # case: 2 <= Ndefect <= 4
        '2' : 100, # case: 5 <= Ndefect <= 10
        '3' : 50, # case: 11 <= Ndefect
        'ds_factor': ds_factor
        }

# For export
var_signif_vals = {
        'Nt_offset' : range_Nt_offset,
        'Ndefect' : range_Ndefect,
        'p_def' : dict_p_def
        }


### Trivial parameters ###
# Defect reflectivity
amp_low = 0.1
# Pulse length
l_pulse_base = 128 # Base pulse length 
fluct_pulse = 10 # Fluctuation of pulse length
# Center frequency fC
fC_base = 3.36* 10**6 #[Hz]
fluct_fC = 0.1 # Relative fluctuation 


#%% Data generation
start_all = time.time()
# Iteration over (1) Nt_offset (2) Ndefect
for depth_setNo, Nt_offset in enumerate(range_Nt_offset):
    # FWM setting
    # Adjust the specimen dimension
    Nt_full = Nt_offset + M
    
    # Specimen parameters
    spc_param = {# Dimension here corresnponds to ROI !!!!!!!
        'dxdata': dx * ureg.meter,
        'dydata': dy * ureg.meter,
        'c0': c0 * ureg.meter / ureg.second,
        'fS': fS * ureg.hertz,
        'Nxdata': Nx,
        'Nydata': Ny,
        'Ntdata': M, # Nt_full - Nt_offset
        'Nxreco': Nx,
        'Nyreco': Ny,
        'Nzreco': M, # Nt_full - Nt_offset
        'anglex': 0 * ureg.degree,
        'angley': 0 * ureg.degree,
        'zImage': -Nt_offset * dz * ureg.meter,
        'xImage': 0 * 0.5 * 10**-3 * ureg.meter,
        'yImage': 0 * 0.5 * 10**-3 * ureg.meter,
        't0': Nt_offset* dt * ureg.second
        }
    # Apodization parameters
    apd_param = {
        'max_angle': 25 * ureg.degree
        }
    
    # Initialize the FWM 
    model = PropagationModel3DSingleMedium()
    model.set_parameters(spc_param)
    model.set_apodizationmodel('Bartlett', apd_param)
    
    for Ndefect in range_Ndefect:
        # Choose the number of datasets according to Ndefect
        setNo = find_element_in_nested_list(dict_p_def['bounds'], Ndefect)[0]
        data_size = dict_p_def[str(setNo)]
        
        ### Base function for data generation with various defect map ###
        def generate_data(counter):
            print('#===========================================================================#')
            print('Dataset: ')
            print('Nt_offset = {} (= No. {}), Ndefect = {}, counter = {}/{}'.format(Nt_offset, depth_setNo, 
                                                                                    Ndefect, int(counter), data_size))
            start = time.time()
            
            # Pulse parameters
            l_pulse = l_pulse_base + np.random.randint(-fluct_pulse, fluct_pulse)
            fC = fC_base* (1 + np.random.uniform(-fluct_fC, fluct_fC)) #[Hz]
            # Set FWM pulse parameters
            pulse_params = {# Pulse parameters
                'pulseLength': l_pulse,
                'fCarrier': fC * ureg.hertz,
                'B': 0.3,
                'fS': fS * ureg.hertz
                }
            model.set_pulsemodel('Gaussian', pulse_params)
            
            # Defect map
            defect_dict = {
                'x': np.arange(Nx),
                'y': np.arange(Ny),
                'z': np.arange(int(0.5* l_pulse), int(M - 0.5* l_pulse)),
                'amp_low' : amp_low,
                'M' : M
                }
            defmap = generate_defect_map(Ndefect, dict_p_def, defect_dict, counter) 
            print('defmap max = {}'.format(defmap.max()))
            
            # Data synthesizer
            synth = DataSynthesizer(model)
            data = synth.get_data(defmap) # shape = M x 1 x 1 x 1 x 1 x Nxdata x Nydata
            data = np.reshape(data.flatten('F'), defmap.shape, 'F') # shape = M x Nxdata x Nydata
            # Normalize
            data = data / np.abs(data).max()
            
            # Save data
            path, fname =  file_export_setting(Nt_offset, Ndefect, int(counter))
            save_data(data, path, fname)
            
            display_time(round(time.time() - start, 3))
            
            del defmap, data
        #######################################################################
        
        np.apply_along_axis(generate_data, 0, np.array([np.arange(data_size)]))
        
        
print('##################################################################')
print('       End of the data generation       ') 
display_time(round(time.time() - start_all, 3))      
print('##################################################################') 
            


