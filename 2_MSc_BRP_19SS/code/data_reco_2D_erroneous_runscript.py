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
   2D Data-Reco Erroneous
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
Nxdata = 20
Ntdata = 680 #880
t0 = 6* 10**-6* ureg.second

c0 = 6300* ureg.metre/ ureg.second #6300
fS = 80* ureg.megahertz #80
openingangle = 25 # curretly does not do anything!

dx = 0.5 # [mm]
dxdata = dx*10**-3* ureg.metre
dzdata = 0.5* c0/(fS.to_base_units())
pos_defect = np.array([[10*dxdata.magnitude, 571*dzdata.magnitude]])* ureg.metre 


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
alpha = 6.25* ureg.megahertz**2 # prameter value from Jan

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


# for error
seed = 5 #5, 0, 3
np.random.seed(seed)


#================================================================================================= p_track setting ===#
# Base of p_scan
p_scan = [] # store all scan positions

# Iteration over scan positions
for idx in range(Nxdata):  
    start = time.time()
    # Set p_true
    curr_p = np.array([idx])*dxdata
    # error
    curr_p = np.array([idx  + 2.52*np.random.normal()])*dxdata
    p_scan.append(round(curr_p[0].magnitude* 10**3, 3))



#=============================================================================================== Position handling ===#
# Save scan positions as tuple
p_scan = tuple(p_scan)
# Quantized scan position
p_qt = np.array(p_scan) 
p_qt = np.around(p_qt/ dx)* dx # unitless, [mm]
p_qt = tuple(p_qt)
# Choose valid entries for reconstruction positions
p_valid = []
posidx_valid = [] # for PosNo
for idx, entry in enumerate(p_qt):
    # Within the valid range?
    if (entry < -0.5) or (entry >= Nxdata* dx):
        pass
    # Repeated entry?
    elif entry in p_valid:
        pass
    else:
        p_valid.append(entry)
        posidx_valid.append(idx)
        
# Save p_reco as tuple
p_valid = tuple(p_valid)
# Sortted version of p_valid for assigning the H into the right position
p_reco = list(p_valid)
p_reco.sort()
# Put PosNo into the corresponding order to p_reco
PosNo_all = []
for element in p_reco:
    # Find the corresnponding index of the p_valid (before sort)
    pidx = p_valid.index(element)
    # Add the corresponding index of p_scan for assigning the Hblock
    PosNo_all.append(posidx_valid[pidx])


#================================================================================== Choose valid entries of A-Scans ===#
# Load Atrue
version = '190815' #'190815', '190815'
Atrue = np.load('npy_data_storage/BScan/BScan_auto_{}.npy'.format(version))

# Base of valid A-Scans
Avalid = np.zeros((Ntdict, Nxdata)) #H = [H_0^{T} H_1^{T} ..... H_{L-1}^{T}], H_i = Ntdict x L

# Select only valid A-Scans
for curr_p, curr_no in zip(p_reco, PosNo_all):
    idx = int(curr_p/dx)
    Avalid[:, idx] = np.array(Atrue[:, curr_no])

plt.figure(1)    
plt.imshow(Avalid)


#======================================================================================== Form complete SAFT matrix ===#
# Base of all SAFT matrix H
H = np.load('npy_data_storage/Hauto_{}/Hauto.npy'.format(version)) #H = [H_0^{T} H_1^{T} ..... H_{L-1}^{T}], H_i = Ntdict x L

start = time.time()

# Reco 
vecA = np.reshape(Avalid, L, order = 'F') # !!! do not forget 'F' !!!
#b = np.dot(np.linalg.pinv(H.T), vecA) # -> does not work, because it is too strict
b = np.dot(H, vecA)
reco = b.reshape(Ntdict, Nxdata, order = 'F')
# save data
# Err = 1mm
#np.save('npy_data_storage/Reco/Reco_track_{}_1mm.npy'.format(version), reco)
# Err = 1.26mm = lambda
#np.save('npy_data_storage/Reco/Reco_track_{}_lambda.npy'.format(version), reco)

plt.figure(2)
plt.imshow(reco)

stop = time.time() - start
print('#======================#')
print('*** Reconsturction ***')
display_time(stop)










