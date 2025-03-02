# -*- coding: utf-8 -*-
"""
Blind position correction via TLS hyperbola fitting and joint optimization
"""

import numpy as np
import matplotlib.pyplot as plt
import time

from blind_error_correction import BlindErrorCorrection2D
from image_quality_analyzer import ImageQualityAnalyzerMSE

          
plt.close('all')   


def display_time(stop):
    if stop > 60 and stop < 60**2:
        print('** Time to complete : {} min'.format(round(stop/60, 2)))
    elif stop >= 60**2:
        print('** Time to complete : {} h'.format(round(stop/(60**2), 2)))
    else:
        print('** Time to complete : {} s'.format(round(stop, 2)))

                                                                                      
#=============================================================================================== Parameter Setting ====#
Nx = 20 # limited due to the opening angle
Nz = 880#200
Nt = Nz
c0 = 6300 #[m/S]
fS = 80*10**6 #[Hz] 
fC = 5*10**6 #[Hz] 
alpha = 20*10**12 #[Hz]**2
dx = 0.5*10**-3 #[m]
dz = 0.5* c0/(fS)
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
# defect position: p_defect
p_def_idx = np.array([10, 571])#91
# For Newton method
Niteration = 15
epsilon = 0.02
#=========================================================================================== True Measurement Data ====#
# Call the error correction class
bec = BlindErrorCorrection2D(fwm_param)
# Generate the true defect map
bec.defectmap_generation(p_def_idx)
defmap_true = bec.defmap_true
# Generate the measurement data (at the each grid position)
bec.data_generation()
a_true, A_true = bec.get_true_data(ret_Atrue = True)
# Position manipulation 
p_true = bec.p_true
K = p_true.shape[0]
# Ref reconstruction: no tracking error
H_true = bec.dictionary_formation(p_true)
reco_true = np.dot(H_true.T, a_true)
Reco_true = np.reshape(reco_true, (Nt, Nx), 'F')
# For calculating SE
analyzer_Ascan = ImageQualityAnalyzerMSE(A_true)
analyzer_Reco = ImageQualityAnalyzerMSE(Reco_true)
del H_true, reco_true

#======================================================================================================= Variables ====#
e_norm = 0.6
e_max = e_norm* wavelength
seed = 23
triangular = False
Njoint = 1 # -> how many times we repeat the joint optimization process
#=========================================================================================== Position Manipulation ====#np.random.seed(seed)
#np.random.seed(seed)
err = np.random.uniform(-e_max, e_max, (K))
x_track = p_true[:, 0] + err # x element
p_track = np.zeros((K, 2))
p_track[:, 0] = x_track
# Model error of the tracked positions
_, A_track = bec.get_true_data(ret_Atrue = True)
se_a_track = analyzer_Ascan.get_mse(A_track)

#====================================================================================== Iterative error correction ====#
#time
start_all = time.time()
# Base to store the data
X_hat = np.zeros((K, Njoint))
deltaX_hat = np.zeros(X_hat.shape)
se_a_opt = np.zeros(Njoint)
Defmap_est = np.zeros((defmap_true.shape[0], Njoint))

# Initialization: x_hat = x_track
x_hat = np.copy(x_track)

for n in range(Njoint):
    # Hyperbola fit
    bec.hyperbola_fit(A_true, x_hat)
    defmap_est = bec.defmap_est
    Defmap_est[:, n] = defmap_est
    # Error correction
    x_hat, deltax_hat, a_opt = bec.error_correction(p_track, a_true, Niteration, epsilon, e_max, triangular)
    X_hat[:, n] = x_hat
    deltaX_hat[:, n] = deltax_hat
    # Evaluation
    A_opt = np.reshape(a_opt, (Nt, K), 'F')
    se_a_opt[n] = analyzer_Ascan.get_mse(A_opt)

# time
print('#=========================================#')
print('End of the error correction:')
end_all = time.time() - start_all
display_time(end_all)  

#================================================================================================== Reconstruction ====#
# Reco with the tracked positios
H_track = bec.dictionary_formation(p_track)
reco_track = np.dot(H_track.T, a_true)
Reco_track = np.reshape(reco_track, (Nt, Nx), 'F')
se_r_track = analyzer_Reco.get_mse(Reco_track)
del H_track, reco_track
# Reco with the corrected positions
p_opt = np.zeros(p_true.shape)
p_opt[:, 0] = x_hat 
H, J = bec.dictionary_formation(p_opt, with_Jacobian = True)
H_opt = H + np.dot(np.kron(np.diag(deltax_hat), np.identity(Nt)), J)
del H, J
reco_opt = np.dot(H_opt.T, a_true)
Reco_opt = np.reshape(reco_opt, (Nt, Nx), 'F')
se_r_opt = analyzer_Reco.get_mse(Reco_opt)
del H_opt, reco_opt
#=========================================================================================================== Plots ====#
def plot_reco(figure_no, data, title):
    plt.figure(figure_no)
    plt.imshow(data)
    plt.title(title)
    plt.xlabel('x / dx')
    plt.ylabel('z / dz')

plot_reco(1, Reco_true, 'reference')
plot_reco(2, Reco_track, 'reco_track w/ the 1st position element')
plot_reco(3, Reco_opt, 'reco w/ the optimized positions')

