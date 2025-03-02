# -*- coding: utf-8 -*-
"""
Runscript for hyperbola fitting w/ TLS
"""
import numpy as np
import matplotlib.pyplot as plt

import hyperbola_fit_TLS

plt.close('all')

# Functions
def get_Gabor_pulse(tof, Ntdata, fS, fCarrier, alpha):
    # reflecting the time shift by ToF
    t = (np.arange(0, Ntdata)/ fS - tof) 
    # Calculate the Gaussian window (i.e. the envelope of the pulse)
    env = np.exp(-alpha* t**2)
    # Calculate the pulse
    return env* np.cos(2* np.pi* fCarrier* t)


def get_tof(x_scan, pos_defect, c0):
    distance = np.sqrt((x_scan - pos_defect[0])**2 + pos_defect[1]**2)
    return 2*distance/ c0


################################################################################################# Parameter setting ####
Nx = 20 # limited due to the opening angle
Nz = 780
Nt = Nz
c0 = 6300 #[m/S]
fS = 80*10**6 #[Hz] 
fCarrier = 5*10**6 #[Hz] 
alpha = 20*10**12 #[Hz]**2
dx = 0.5*10**-3 #[m]
dz = 0.5* c0/(fS)
wavelength = 1.26* 10**-3 # [m]
pos_defect = np.array([10*dx, 671*dz]) #571*dz
Nt_offset = 580


################################################################################################### Data generation ####
pos_scan_idx = np.arange(Nx)#np.array([3, 5, 11]) #index![5, 8, 13]
pos_scan = pos_scan_idx* dx
# Base for the measurment data
meas_data = np.zeros((Nt, len(pos_scan)))

for curr_col in range(meas_data.shape[1]):
    curr_xscan = pos_scan[curr_col]
    curr_tof = get_tof(curr_xscan, pos_defect, c0)
    curr_ascan = get_Gabor_pulse(curr_tof, Nt, fS, fCarrier, alpha)
    meas_data[:, curr_col] = curr_ascan  
 
bscan = np.zeros((Nt, Nx))
bscan[:, pos_scan_idx] = meas_data # assign measurement data into B-Scan

############################################################################################# Position manipulation ####
e_max_norm = 0.5
e_max = e_max_norm* wavelength
seed = 72
# Initialization of np.random
np.random.seed(seed)
# Tracking error w/ the normal distribution
err_track = np.random.uniform(-e_max, e_max)
#np.random.seed(seed)
#err_track = np.random.uniform(0, 2*sigma, pos_scan.shape[0])
x_track = pos_scan + err_track

################################################################################################# Hyperbola fitting ####
hypfit = hyperbola_fit_TLS.HyperbolaFitTLS2D(dz, Nt_offset)
hypfit.find_peaks(bscan[Nt_offset:, :])
hypfit.solve_TLS(x_track)
xdef_estTLS, zdef_estTLS = hypfit.x_def, hypfit.z_def
# Defect map conversion
hypfit.convert_to_defect_map(Nx, Nz, dx)
defmap_est = hypfit.get_defect_map()
Defmap_est = np.reshape(defmap_est, ((Nt - Nt_offset), Nx), 'F')
#Defmap_est = np.concatenate((np.zeros((Nt_offset, Nx)), Defmap_est))

print('#===== Data info =====#')
print('dx: {}mm, dz = {}mm, lambda = {}mm'.format(dx*10**3, dz*10**3, wavelength*10**3))
print('x_def: {}mm, z_def = {}mm'.format(pos_defect[0]*10**3, pos_defect[1]*10**3))
print('Tracking error w/ e_max = {}* lambda'.format(e_max))      
print('#===== Estimation error via TLS =====#')
print('x_def: {}* dx, z_def = {}* dz'.format(round((pos_defect[0] - xdef_estTLS)/dx, 4), 
                                         round((pos_defect[1] - zdef_estTLS)/dz, 4)))
  
#plt.imshow(Defmap_est[580:, :])
plt.imshow(Defmap_est)