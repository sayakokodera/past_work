# -*- coding: utf-8 -*-
import numpy as np
import json
import matplotlib.pyplot as plt
import time

from tools.datetime_formatter import DateTimeFormatter
from ultrasonic_imaging_python.definitions import units
ureg = units.ureg

from ultrasonic_imaging_python.forward_models.data_synthesizers_progressive import DataGeneratorProgressive2D
from ultrasonic_imaging_python.sako_tools.image_quality_analyzer import ImageQualityAnalyzerMSE
from ultrasonic_imaging_python.sako_tools.parameter_dictionary_exporter import ParameterDictionaryExporter

"""
#===========================#
   2D Data-Reco Reference
#===========================#
What this script does:
    (0) Parameter setting
    (1) Get A-Scans through forming H
        vec{A} = H^{T} \cdot b
            A = array with Nt x Nx, containing all A-Scans (i.e. B-Scan)
            H = array with L x L (L = Nt* Nx), collection of all SAFT matrix at each scan position
                H_i = i-th SAFT matrix with Nt x L, then
                H = [H_0^{T} H_1^{T} ..... H_{L-1}^{T}]
                -> all A-Scans for one possible defect position = row vector of H! 
            b = vec with the sieze L, vectorized defect map
    (2) Check H \cdot H^{T}?
    (3) Reco
"""
#======================================================================================================= Functions ===#

def display_time(stop):
    if stop > 60 and stop < 60**2:
        print('Time to complete : {} min'.format(round(stop/60, 2)))
    elif stop >= 60**2:
        print('Time to complete : {} h'.format(round(stop/(60**2), 2)))
    else:
        print('Time to complete : {} s'.format(round(stop, 2)))


#================================================================================================ Parameter Setting ===#
### Specimen params ###
Nxdata = 11
Ntdata = 680 #880
t0 = 6* 10**-6* ureg.second

c0 = 6300* ureg.metre/ ureg.second #6300
fS = 80* ureg.megahertz #80
openingangle = 25 # curretly does not do anything!

dx = 1.25 # [mm]
dxdata = dx*10**-3* ureg.metre
dzdata = 0.5* c0/(fS.to_base_units())
pos_defect = np.array([[10*dxdata.magnitude, 571*dzdata.magnitude]])* ureg.metre 
pos_defect = np.array([[5* 10**-3, 571*dzdata.magnitude]])* ureg.metre 


specimen_params = {
        'Ntdata' : Ntdata,
        'Nxdata' : Nxdata,
        'c0' : c0,
        'fS' : fS,
        'dxdata' : dxdata,
        'dzdata' : dzdata,
        'openingangle' : openingangle,
        'pos_defect' : pos_defect,
        't0' : t0,
        }


### Pulse params ###
fCarrier = 5* ureg.megahertz 
alpha = 30* ureg.megahertz**2 # 6.25 prameter value from Jan

# Pulse setting for direct Gabor pulse calculation without refmatrix
pulse_params = {
        'fCarrier' : fCarrier,
        'fS' : fS,
        'alpha' : alpha
        }

# Stepsize for the SAFT dictionary 
stepsize = dxdata  #with unit

# for B-Scan generation -> the size of the ROI is required
Ntdict = Ntdata - int(t0 / ((1/fS).to_base_units()))
# for Reco
L = Ntdict* Nxdata

# Base of fname to store each SAFT matrix H_i
fname_base = 'npy_data_storage/Hauto/saft_matrix_PosNo{}.npy'


#================================================================================================ A-Scan Generation ===#
# Base of all A-Scans, i.e. B-Scan
A = np.zeros((Ntdict, Nxdata))
Adelta = np.zeros((Ntdict, Nxdata))
p_scan = [] # store all scan positions

# Iteration over scan positions
for idx in range(Nxdata):  
    print('*** Iteration No. {} ***'.format(idx))
    start = time.time()
    # Set p_true
    curr_p = np.array([idx])*dxdata
    p_scan.append(round(curr_p[0].magnitude* 10**3, 3))
    # Call FWM
    fwm_prog = DataGeneratorProgressive2D(specimen_params, stepsize)
    fwm_prog.set_pulse_parameter(pulse_params)
    fwm_prog.unit_handling_specimen_params()
    fwm_prog.vectorize_defect_map()
    fwm_prog.set_reflectivity_matrix(curr_p, oracle_cheating = False)
    # Calculte SAFT matrix for current p
    Hblock = fwm_prog.get_impulse_response_dictionary(without_refmatrix = True)
    # Get A-Scan
    A[:, idx] = fwm_prog.get_single_ascan(Hblock)
    Adelta[:, idx] = fwm_prog.get_single_ascan(fwm_prog.refmatrix)
    # save Htrue -> fatser calculation
    fname = fname_base.format(idx)
    np.save(fname, Hblock)
    del Hblock
    
    stop = time.time() - start
    display_time(stop)


plt.figure(1)    
plt.imshow(Adelta)

plt.figure(2)    
plt.imshow(A)

#============================================================================================= Complete SAFT matrix ===#
# Base of all SAFT matrix H
H = np.zeros((L, L)) #H = [H_0^{T} H_1^{T} ..... H_{L-1}^{T}], H_i = Ntdict x L

start = time.time()
# Load data through iteration over p
for p_idx in range(len(p_scan)):
    # load the current SAFT matrix
    curr_H = np.load(fname_base.format(p_idx))
    # Set the start&end for the current ditionary block
    block_start = p_idx* curr_H.shape[0]
    block_end = (p_idx + 1)* curr_H.shape[0]
    # Put the current dict into SAFT matrix
    H[:, block_start:block_end] = np.array(curr_H.T)


# Reco 1
vecA = np.reshape(A, L, order = 'F') # !!! do not forget 'F' !!!
b1 = np.dot(H, vecA) # -> does not work!
plt.figure(3)
plt.imshow(b1.reshape(Ntdict, Nxdata, order = 'F'))

# Reco2
#b2 = np.dot(np.linalg.pinv(H.T), vecA) #-> works, data saved (190813)
#plt.figure(4)
#plt.imshow(b2.reshape(Ntdict, Nxdata, order = 'F'))

stop = time.time() - start
print('#======================#')
print('*** Reconsturction ***')
display_time(stop)













