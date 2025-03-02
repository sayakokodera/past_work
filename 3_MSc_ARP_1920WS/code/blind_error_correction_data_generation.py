"""
TLS + Newton error correction: runscript with parameter changes
"""
import numpy as np
import matplotlib.pyplot as plt
import time

from blind_error_correction import BlindErrorCorrection2D
from tools.npy_file_writer import save_data
from tools.datetime_formatter import DateTimeFormatter 

"""
What this script should do:
    - Load the fixed variables from the given json files (or set the fixed param values here)
    - Set the variables to the given variable values
        e_norm, z_idx, Nrealization
    *** Iteration over z_idx ***
    - Generate A_true
    - Calculate Reco_true (for reference) -> save, delete
    *** Iteration over e_norm ***
    *** Iteration over Nrealization ***
    - Position manipulation 
    - Calculate Reco_track  -> save, delete
    - Run BlindErrorCorrection2D 
    - Calculate Reco_opt -> save, delete
    *** end realNo ***
    - Calculate the average estimation error = np.abs(x_true - x_opt) -> save

In the npy_data folder the following data should be saved:
    - npy_data/today/z_def_mm/Reco_true.npy (x_range: x_min = x_idx - 10, xmax = x_idx + 10)
    - npy_data/today/z_def_mm/e_norm/Reco_track/realNo.npy
    - npy_data/today/z_def_mm/e_norm/Reco_opt/realNo.npy
    - npy_data/today/z_def_mm/e_norm/ME_xopt.npy
"""
plt.close('all')
#======================================================================================================= Functions ====#
def calculate_dimensions(z_def_mm, dz, opening_angle, dx, Nzimg):
    z_idx = int((z_def_mm* 10**-3)/dz)
    Nz = z_idx + int(Nzimg/2)
    Nx = int(2* (Nz*dz*np.tan(np.deg2rad(0.5*opening_angle)))/dx) + 5 #- 10
    if Nx < 21:
        Nx = 21
    elif Nx > 50:
        Nx = Nx - 10
    x_idx = int(Nx/2)
    return Nx, Nz, x_idx, z_idx


def calculate_ROI(Nt, Nzimg):
    Nt_offset = max(0, Nt - Nzimg)
    M = Nt - Nt_offset
    return Nt_offset, M


def calculate_Reco(bec, p, with_Jacobian = False, deltax = None):
    M = bec.M
    Nx = bec.Nx
    # Without spatial approximation (for H_true and H_track)
    if with_Jacobian == False: 
        H = bec.dictionary_formation(p)
    # With spatial approximation (for H_opt)
    else:
        H_base, J_base = bec.dictionary_formation(p, with_Jacobian = True)
        H = H_base + np.dot(np.kron(np.diag(deltax), np.identity(M)), J_base)
        del H_base, J_base
    reco = np.dot(H.T, bec.a_true)
    Reco = np.reshape(reco, (M, Nx), 'F')
    del H
    return Reco


def save_Reco(reco, Nx, path, fname):
    if Nx < 21:
        data = np.copy(reco)
    else: # save only the region of the size M x 21 
        x_idx = int(Nx/2)
        x_min = x_idx - 10
        x_max = x_min + 21
        data = np.copy(reco[:, x_min:x_max])
    save_data(data, path, fname) #-> add comment in save_data as "Data saved!"
    del reco, data
    
        
def plot_reco(figure_no, data, title, dx, dz):
    fig = plt.figure(figure_no)
    ax = fig.add_subplot(111)
    ax.imshow(data)
    ax.set_aspect(dz/dx)
    ax.set_title(title)
    ax.set(xlabel = 'x / dx', ylabel = 'z / dz')
    

def position_manipulation(e_max, p_true, triangular = False):
    if triangular == True:
        err = np.random.triangular(-e_max, 0, e_max)
    else:
        err = np.random.uniform(-e_max, e_max, (len(p_true)))
    p_track = np.copy(p_true)
    p_track[:, 0] = p_track[:, 0] + err 
    return p_track


def tracking_error_correction(bec, p_track, Niteration, epsilon, e_max, triangular):
    # Base setting
    a_true, A_true = bec.get_true_data(ret_Atrue = True)
    # Hyperbola fit -> estimate the defect position
    bec.hyperbola_fit(A_true, p_track[:, 0])
    # Error correction
    x_opt, deltax_opt, _ = bec.error_correction(p_track, a_true, Niteration, epsilon, e_max, triangular)
    # x_opt -> p_opt
    p_opt = np.zeros(p_track.shape)
    p_opt[:, 0] = x_opt
    return p_opt, deltax_opt


def display_time(stop):
    if stop > 60 and stop < 60**2:
        print('** Time to complete : {} min'.format(round(stop/60, 2)))
    elif stop >= 60**2:
        print('** Time to complete : {} h'.format(round(stop/(60**2), 2)))
    else:
        print('** Time to complete : {} s'.format(round(stop, 2)))


def create_path(data_type, today, z_def_mm, triangular = None, e_norm = None):
    """ e.g. 
        npy_data/today/z_def_mm/Reco_true.npy
        npy_data/today/z_def_mm/e_norm/Reco_track/realNo.npy
        npy_data/today/z_def_mm/e_norm/Reco_opt/realNo.npy
        npy_data/today/z_def_mm/e_norm/ME_xopt.npy
    """
    if data_type == 'Reco_true':
        path = 'npy_data/{}/{}mm'.format(today, z_def_mm)
    else:
        if triangular == False:
            if data_type == 'me_xopt':
                path = 'npy_data/{}/{}mm/{}lambda'.format(today, z_def_mm, e_norm)
            else:
                path = 'npy_data/{}/{}mm/{}lambda/{}'.format(today, z_def_mm, e_norm, data_type)
        else:
            if data_type == 'me_xopt':
                path = 'npy_data/{}/{}mm_triangular/{}lambda'.format(today, z_def_mm, e_norm)
            else:
                path = 'npy_data/{}/{}mm_triangular/{}lambda/{}'.format(today, z_def_mm, e_norm, data_type)
    return path

#=============================================================================================== Parameter Setting ====#
opening_angle = 37 #[deg]
c0 = 6300 #[m/S]
fS = 80*10**6 #[Hz] 
fC = 5*10**6 #[Hz] 
alpha = 20*10**12 #[Hz]**2
dx = 0.5*10**-3 #[m]
dz = 0.5* c0/(fS) 
wavelength = 1.26* 10**-3 # [m]  
Nzimg = 200 

# Get date for saving data
dtf = DateTimeFormatter()
today = dtf.get_date_str()
today = '200401'
#======================================================================================================= Variables ====#
# Main variables
z_def_mm_all = np.array([30])# [mm]
err_norm_all = np.array([1])#np.around(np.arange(0.1, 0.901, 0.1), 2)# Bound for the tracking error, normaliyed with the wavelength

Nrealization_default = 1
buffer = 0
# Error distribution
triangular = False # if False, use uniform distribution
# Newton method variables
Niteration = 10
epsilon = 0.02

#======================================================================================================================#
# time
start_all = time.time()

for z_def_mm in z_def_mm_all:
    plt.close('all')
    # time
    start_zdef = time.time()
    
    # Set dimensions and the rest of the parameters
    Nx, Nz, x_idx, z_idx = calculate_dimensions(z_def_mm, dz, opening_angle, dx, Nzimg)
    Nt_offset, M = calculate_ROI(Nz, Nzimg)
    p_def_idx = np.array([x_idx, z_idx])
    fwm_param = {'Nx' : Nx, 'Nz' : Nz, 'Nt' : Nz, 'c0' : c0, 'fS' : fS, 'fC' : fC, 'alpha' : alpha, 'dx' : dx,
                 'opening_angle' : opening_angle}
    
    # Call the error correction class
    bec = BlindErrorCorrection2D(fwm_param, Nt_offset)
    
    # Generate measurement data (at the each grid position)
    bec.defectmap_generation(p_def_idx)
    bec.data_generation()
    p_true = bec.p_true
    
# =============================================================================
#     # Reco_true
#     start_recotrue = time.time()
#     Reco_true = calculate_Reco(bec, p_true)
#     path_Reco_true = create_path('Reco_true', today, z_def_mm)
#     save_Reco(Reco_true, Nx, path_Reco_true, 'Reco_true_largeROI.npy')
#     plot_reco(1, Reco_true, 'Reco_true', dx, dz) 
#     del Reco_true
#     print('Reco_true calculation:')
#     display_time(time.time() - start_recotrue)
#     
# =============================================================================
# =============================================================================
#     for idx, e_norm in enumerate(err_norm_all):
#         if e_norm == 0:
#             Nrealization = 1
#         else:
#             Nrealization = Nrealization_default
#         start_err = time.time()
#         e_max = e_norm* wavelength
#         
#         # For calculating the estimation error of x_opt
#         me_xopt = np.zeros(p_true.shape[0])
#         
#         for realNo in np.arange(buffer, Nrealization + buffer):
#             start = time.time()
#             print('#============================================#')
#             print('z_def = {}mm, error {} lambda, realization {}'.format(z_def_mm, e_norm, realNo))
#             print('#============================================#')
#             
#             # Position manipulation
#             p_track = position_manipulation(e_max, p_true, triangular = False)
#             me_track = np.abs(p_true[:, 0] - p_track[:, 0])/wavelength
#             
#             # Reco_track
#             Reco_track = calculate_Reco(bec, p_track)
#             path_Reco_track = create_path('Reco_track', today, z_def_mm, triangular, e_norm)
#             save_Reco(Reco_track, Nx, path_Reco_track, 'No.{}.npy'.format(realNo))
#             plot_reco(2, Reco_track, 'Reco_track', dx, dz)
#             del Reco_track
#             
#             # Reco_opt
#             p_opt, deltax_opt = tracking_error_correction(bec, p_track, Niteration, epsilon, e_max, triangular)
#             Reco_opt = calculate_Reco(bec, p_opt, with_Jacobian = True, deltax = deltax_opt)
#             path_Reco_opt = create_path('Reco_opt', today, z_def_mm, triangular, e_norm)
#             save_Reco(Reco_opt, Nx, path_Reco_opt, 'No.{}.npy'.format(realNo))
#             plot_reco(3, Reco_opt, 'Reco_opt', dx, dz)   
#             del Reco_opt
#             
#             # Estimation error of x_opt
#             me_xopt += np.abs(p_true[:, 0] - p_opt[:, 0])
#             
#             # time
#             print('End of realization No.{}'.format(realNo))
#             stop = time.time() - start
#             display_time(stop)
#             
#         # Calculate the mean error -> save
#         me_xopt = (me_xopt / Nrealization)/wavelength
#         path_me_xopt = create_path('me_xopt', today, z_def_mm, triangular, e_norm)
#         save_data(me_xopt, path_me_xopt, 'me_xopt.npy')
#             
#         # time
#         print('#=========================================#')
#         print('End of error {} lambda'.format(e_norm))
#         stop_err = time.time() - start_err
#         display_time(stop_err)
#         
#     # time
#     print('#=========================================#')
#     print('End of z_def = {}mm'.format(z_def_mm))
#     stop_zdef = time.time() - start_zdef
#     display_time(stop_zdef)
#             
# # time
# now = dtf.get_time_str()
# print('#=========================================#')
# print('Total computation time:')
# end_all = time.time() - start_all
# display_time(end_all)  
# print('Finished at {}'.format(now))          
# =============================================================================
           
            
# Plots
# =============================================================================
# plot_reco(1, Reco_true, 'Reco_true', dx, dz)            
# plot_reco(2, Reco_track, 'Reco_track', dx, dz)
# plot_reco(3, Reco_opt, 'Reco_opt', dx, dz)    
# =============================================================================
    
    

