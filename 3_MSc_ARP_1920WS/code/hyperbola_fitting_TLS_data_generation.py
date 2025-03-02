# -*- coding: utf-8 -*-
"""
Hyperbola fitting via TLS: data generation
"""
import numpy as np
import matplotlib.pyplot as plt
import time

from blind_error_correction import BlindErrorCorrection2D
from hyperbola_fit_TLS import HyperbolaFitTLS2D
from tools.npy_file_writer import save_data
from tools.datetime_formatter import DateTimeFormatter 


def display_time(stop):
    if stop > 60 and stop < 60**2:
        print('** Time to complete : {} min'.format(round(stop/60, 2)))
    elif stop >= 60**2:
        print('** Time to complete : {} h'.format(round(stop/(60**2), 2)))
    else:
        print('** Time to complete : {} s'.format(round(stop, 2)))


def create_path(today, triangular):    
    if triangular == False:
        path = 'npy_data/{}/{}'.format(today, 'uniform')
    else:
        path = 'npy_data/{}/{}'.format(today, 'triangular')
    return path


#======================================================================================================= Variables ====#
Nrealization = 5000
triangular = False # if False, use uniform distribution
err_max_norm = np.around(np.arange(0.0, 1.00001, 0.01), 3) # Bound for the tracking error [m], np.around(np.arange(0.0, 0.176, 0.025), 3)

# defect position: p_defect
z_idx = 1270 # 571dz = 22.5mm, 762dz = 30mm, 1270dz = 50mm, 1905*dz = 75mm
# Computational time for 10 iterations:
# 571 -> 1.5min, 762 -> 2.3 min, 1270 -> 9min
#=============================================================================================== Parameter Setting ====#
openingangle = 25 #[deg]
c0 = 6300 #[m/S]
fS = 80*10**6 #[Hz] 
fC = 5*10**6 #[Hz] 
alpha = 20*10**12 #[Hz]**2
dx = 0.5*10**-3 #[m]
dz = 0.5* c0/(fS)
# Change the size of the specimen according to our ROI, i.e. defect depth
Nz = z_idx + 100
Nt = Nz
# Nx should be varied acccording to the opening angle and our ROI
Nx = int(2* (z_idx*dz*np.tan(np.deg2rad(12.5)))/dx)
x_idx = int(Nx/2)
fwm_param = {
    'Nx' : Nx,
    'Nz' : Nz,
    'Nt' : Nt,
    'c0' : c0, 
    'fS' : fS, 
    'fC' : fC, 
    'alpha' : alpha,
    'dx' : dx
        }
wavelength = 1.26* 10**-3 # [m]
Nt_offset = max(0, Nz - 200)
M = Nt - Nt_offset
# defect position: p_defect
p_def_idx = np.array([x_idx, z_idx])
xdef_true = p_def_idx[0]*dx #[m]
zdef_true = p_def_idx[1]*dz #[m]
p_def_true = np.array([xdef_true, zdef_true])

# Setting for saving data
dtf = DateTimeFormatter()
today = dtf.get_date_str()
now = dtf.get_time_str()

#=========================================================================================== True Measurement Data ====#
# Call the error correction class
bec = BlindErrorCorrection2D(fwm_param, Nt_offset)
# Generate the true defect map
bec.defectmap_generation(p_def_idx)
defmap_true = bec.defmap_true
# Generate the measurement data (at the each grid position)
bec.data_generation()
a_true, A_true = bec.get_true_data(ret_Atrue = True)
# Position manipulation 
p_true = bec.p_true
# delete unneccesary data
del defmap_true

#================================================================================================ Error Correction ====#
# Dimensions
K = p_true.shape[0] # number of scan positions
e_defmap_est = np.zeros((Nrealization, err_max_norm.shape[0]))
# time
start_all = time.time()

for idx, e_norm in enumerate(err_max_norm):
    # time
    start_err = time.time()
    e_max = e_norm* wavelength    
    
    for realNo in range(Nrealization):        
        # Position manipulation
        if triangular == True:
            err = np.random.triangular(-e_max, 0, e_max)
        else:
            err = np.random.uniform(-e_max, e_max, (K))
        x_track = p_true[:, 0] + err # x element
        p_track = np.zeros((K, 2))
        p_track[:, 0] = x_track
        
        # Hyperbola fit
        hypfit = HyperbolaFitTLS2D(dz, Nt_offset)
        hypfit.find_peaks(A_true)
        hypfit.solve_TLS(x_track)
        xdef_est, zdef_est = hypfit.x_def, hypfit.z_def
        # Estimation error
        p_def_est = np.array([xdef_est, zdef_est])
        e_defmap_est[realNo, idx] = np.linalg.norm((p_def_true - p_def_est)) #[m]
        
    # time
    print('#=========================================#')
    print('End of Error No.{}'.format(idx))
    stop_err = time.time() - start_err
    display_time(stop_err)

# time
print('#=========================================#')
print('End of all error:')
end_all = time.time() - start_all
display_time(end_all)
        
        
#======================================================================================================= Save Data ====#      
# For saving data
e_est = np.mean(e_defmap_est, axis = 0)
del e_defmap_est
path = create_path(today, triangular)
data = np.stack((err_max_norm, e_est)).T
save_data(data, path, 'defect_estimation_error_{}dz.npy'.format(z_idx))


plt.figure(6)
plt.plot(err_max_norm, e_est*10**3, label = '50mm')
#plt.plot(err_max_norm, e_est762*10**3, label = '30mm')
plt.title('Estimation error [mm]')
plt.xlabel('Tracking error / lambda')
plt.ylabel('p_def - \hat{p_def}')
