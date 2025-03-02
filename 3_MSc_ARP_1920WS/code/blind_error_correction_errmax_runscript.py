# -*- coding: utf-8 -*-
"""
Error Correction w/ err_max
"""
import numpy as np
import matplotlib.pyplot as plt
import time

from blind_error_correction import BlindErrorCorrection2D
from tools.npy_file_writer import save_data
from tools.datetime_formatter import DateTimeFormatter 
from image_quality_analyzer import ImageQualityAnalyzerMSE
from visualization_2D_runscript import savepng


plt.close('all')   

def display_time(stop):
    if stop > 60 and stop < 60**2:
        print('** Time to complete : {} min'.format(round(stop/60, 2)))
    elif stop >= 60**2:
        print('** Time to complete : {} h'.format(round(stop/(60**2), 2)))
    else:
        print('** Time to complete : {} s'.format(round(stop, 2)))


def create_path(today, curr_err_max, z_def_idx, triangular):
    if curr_err_max < 0.1:
        err_str = '00{}'.format(int(curr_err_max*10**3))
    elif curr_err_max < 1:
        err_str = '0{}'.format(int(curr_err_max*10**3))
    else:
        err_str = '{}'.format(int(curr_err_max*10**3))
    
    if triangular == False:
        path = 'npy_data/{}/{}_{}dz/{}_lambda'.format(today, 'uniform', z_def_idx, err_str)
    else:
        path = 'npy_data/{}/{}_{}dz/{}_lambda'.format(today, 'triangular', z_def_idx,err_str)
    return path


#======================================================================================================= Variables ====#
Nrealization_default = 200
triangular = False # if False, use uniform distribution
err_max_norm = np.around(np.arange(0.0, 1.001, 0.1), 2) # Bound for the tracking error [m], np.around(np.arange(0.0, 0.176, 0.025), 3)

# defect position: p_defect
z_idx = 762 # 571dz = 22.5mm, 762dz = 30mm, 1270dz = 50mm, 1905*dz = 75mm
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
p_def_idx = np.array([x_idx, z_idx])
 
# For Newton method
Niteration = 10
epsilon = 0.02

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
# Ref reconstruction: no tracking error
H_true = bec.dictionary_formation(p_true)
reco_true = np.dot(H_true.T, a_true)
Reco_true = np.reshape(reco_true, (M, Nx), 'F')
# For calculating SE
analyzer_Ascan = ImageQualityAnalyzerMSE(A_true)
analyzer_Reco = ImageQualityAnalyzerMSE(Reco_true)
del H_true, reco_true
# Save the reco data as png
#savepng(Reco_true, z_idx, 'reco_true', 0, 100, x_idx)
#================================================================================================ Error Correction ====#
# Dimensions
K = p_true.shape[0] # number of scan positions

# Average estimation/modeling error -> for evaluation
mse_xhat = np.zeros((err_max_norm.shape[0], 2)) # Mean squared error of the optimized positions
mse_xhat[:, 0] = np.copy(err_max_norm)
mse_recotrack = np.copy(mse_xhat)
mse_recoopt = np.copy(mse_xhat)

# time
start_all = time.time()

for idx, e_norm in enumerate(err_max_norm):
    if e_norm == 0:
        Nrealization = 1
    else:
        Nrealization = Nrealization_default
    start_err = time.time()
    e_max = e_norm* wavelength
    # For saving data
    path = create_path(today, e_norm, p_def_idx[1], triangular)
    X_track = np.zeros((K, Nrealization))
    X_hat = np.zeros((K, Nrealization)) # Position estimation through correcting the error
    deltaX_hat = np.zeros((K, Nrealization)) # Estimated & corrected tracking error
    se_a_track = np.zeros(Nrealization)
    se_a_opt = np.zeros(Nrealization)
    se_reco_track = np.zeros(Nrealization)
    se_reco_opt = np.zeros(Nrealization)
    
    for realNo in range(Nrealization):
        start = time.time()
        print('#=========================================#')
        print('Error No.{}, realization {}'.format(idx, realNo))
        print('#=========================================#')
              
        # Position manipulation
        if triangular == True:
            err = np.random.triangular(-e_max, 0, e_max)
        else:
            #np.random.seed(71)
            err = np.random.uniform(-e_max, e_max, (K))
        x_track = p_true[:, 0] + err # x element
        #x_track[11] = p_def_idx[0]*dx # manipulate the tracked position directly above the defect
        p_track = np.zeros((K, 2))
        p_track[:, 0] = x_track
        X_track[:, realNo] = x_track
        print('Error directly above the defect: {}mm'.format((p_true[p_def_idx[0], 0] - x_track[p_def_idx[0]])*10**3))
        
        # Hyperbola fit
        bec.hyperbola_fit(A_true, x_track)
        defmap_est = bec.defmap_est
        # Error correction
        x_hat, deltax_hat, a_opt = bec.error_correction(p_track, a_true, Niteration, epsilon, e_max, triangular)
        X_hat[:, realNo] = x_hat
        deltaX_hat[:, realNo] = deltax_hat
        
        #### Evaluation ###
        # With error correction
        A_opt = np.reshape(a_opt, (M, K), 'F')
        se_a_opt[realNo] = analyzer_Ascan.get_mse(A_opt)
        del A_opt
        p_opt = np.zeros((K, 2))
        p_opt[:, 0] = x_hat 
        H, J = bec.dictionary_formation(p_opt, with_Jacobian = True)
        H_opt = H + np.dot(np.kron(np.diag(deltax_hat), np.identity(M)), J)
        del H, J
        reco_opt = np.dot(H_opt.T, a_true)
        Reco_opt = np.reshape(reco_opt, (M, Nx), 'F')
        se_reco_opt[realNo] = analyzer_Reco.get_mse(Reco_opt)
        del H_opt, reco_opt#, Reco_opt
       
        # Without error correction
        H_track = bec.dictionary_formation(p_track)
        A_track = np.reshape(np.dot(H_track, defmap_est), (M, K), 'F')
        se_a_track[realNo] = analyzer_Ascan.get_mse(A_track)
        del A_track
        reco_track = np.dot(H_track.T, a_true)
        Reco_track = np.reshape(reco_track, (M, Nx), 'F')
        se_reco_track[realNo] = analyzer_Reco.get_mse(Reco_track)
        del H_track, reco_track#, Reco_track
    
    # Save data -> to reproduce the situation
    save_data(X_track, path, 'X_track.npy')
    save_data(X_hat, path, 'X_opt.npy')
    save_data(deltaX_hat, path, 'deltaX_opt.npy')
    save_data(se_a_track, path, 'se_a_track.npy')
    save_data(se_a_opt, path, 'se_a_opt.npy')
    save_data(se_reco_track, path, 'se_reco_track.npy')
    save_data(se_reco_opt, path, 'se_reco_opt.npy')
    
    # Save reco arr as png
    if e_norm - int(e_norm) == 0 or e_norm - int(e_norm) == 0.5:
        savepng(Reco_opt, z_idx, 'reco_opt', e_norm, 100, x_idx)
        savepng(Reco_track, z_idx, 'reco_track', e_norm, 100, x_idx)
    
    # Evaluation
    e_xhat = np.mean((np.repeat(np.array([p_true[:, 0]]).T, Nrealization, axis = 1) - X_hat), axis = 1)
    mse_xhat[idx, 1] = np.mean(e_xhat)*10**3 #[mm]
    mse_recotrack[idx, 1] = np.mean(se_reco_track)
    mse_recoopt[idx, 1] = np.mean(se_reco_opt)
    
    
    # time
    print('#=========================================#')
    print('End of Error No.{}'.format(idx))
    stop_err = time.time() - start_err
    display_time(stop_err)
    
# time
print('#=========================================#')
print('End of the error correction:')
end_all = time.time() - start_all
display_time(end_all)

# Save data for evaluation
path = 'npy_data/{}/{}_{}dz'.format(today, 'uniform', p_def_idx[1])
save_data(mse_xhat, path, 'mse_xhat.npy')
save_data(mse_recotrack, path, 'mse_recotrack.npy')
save_data(mse_recoopt, path, 'mse_recoopt.npy')


#=========================================================================================================== Plots ====#
def plot_reco(figure_no, data, title):
    plt.figure(figure_no)
    plt.imshow(data)
    plt.title(title)
    plt.xlabel('x / dx')
    plt.ylabel('z / dz')


plot_reco(1, Reco_true, 'reco true')
plot_reco(2, Reco_track, 'reco track: uncorrected')
plot_reco(3, Reco_opt, 'reco opt: corrected')

