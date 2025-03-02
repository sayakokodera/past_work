# -*- coding: utf-8 -*-
"""
MUSE reconstruction
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import scipy.signal as scsi
import time

from tof_calculator import ToFforDictionary2D
from dictionary_former import DictionaryFormer
from blind_error_correction import BlindErrorCorrection2D
from display_time import display_time
from signal_denoising import denoise_svd

plt.close('all') 

#======================================================================================================= Functions ====#
def get_opening_angle(c0, D, fC):
    r""" The opening angle (= beam spread), theta [grad], of a trnsducer element can be calculated with
        np.sin(theta) = 1.2* c0 / (D* fC) with
            c0: speed of sound [m/S]
            D: element diameter [m]
            fC: career frequency [Hz]
    (Cf: https://www.nde-ed.org/EducationResources/CommunityCollege/Ultrasonics/EquipmentTrans/beamspread.htm#:~:text=Beam%20spread%20is%20a%20measure,is%20twice%20the%20beam%20divergence.)
    """
    theta_rad = np.arcsin(1.2* c0/ (D* fC))
    return np.rad2deg(theta_rad) 


def plot_reco(figure_no, data, title, dx, dz):
    fig = plt.figure(figure_no)
    ax = fig.add_subplot(111)
    ax.imshow(data)
    ax.set_aspect(dz/dx)
    ax.set_title(title)
    ax.set(xlabel = 'x / dx', ylabel = 'z / dz')
#============================================================================================== Parameter Setting =====#   
# ROI
Nt_offset, Nz = 1900, 2000 #2025, 2175# = Nt_offset, Nt
xmin, xmax = 185, 225#20, 100
y = 289
M = Nz - Nt_offset
Nx = xmax - xmin
dx = 0.5*10**-3 #[m]
dz = 3.6875*10**-5 #[m]
# Measurement setting
c0 = 5900 #[m/S]
fS = 80*10**6 #[Hz]  
fC = 3.36*10**6 #[Hz] 
D = 10*10**-3 #[m], transducer element diameter
opening_angle = 170#get_opening_angle(c0, D, fC)
# Model parameter (estimated from the measurement data)
alpha = 10.6*10**12 #[Hz]**2, bandwidth factor
r = 0.75 # Chirp rate
with_envelope = False
#====================================================================================================== MUSE data =====#
# load MUSE
data_all = scio.loadmat('MUSE/measurement_data.mat')
data = data_all['data'][Nt_offset:Nz, y, xmin:xmax]
data = data / np.abs(data).max()
cscan = np.max(data_all['data'], axis = 0)
del data_all

#%% SAFT Reco
# ToF calculation
start_tof = time.time()
p_scan = np.zeros((Nx, 2))
p_scan[:, 0] = np.arange(Nx)* dx
tofcalc = ToFforDictionary2D(c0, Nx, Nz, dx, dz, p_scan, Nt_offset, opening_angle)
tofcalc.calculate_tof(calc_grad = False)
tof = tofcalc.get_tof()
apf = tofcalc.get_apodization_factor() # Apodization factors
del tofcalc
print('ToF = {}S'.format(round(time.time() - start_tof, 3)))

# Dictionary formation
start_dict = time.time()
dformer = DictionaryFormer(Nz, fS, fC, alpha, Nt_offset, r) 
dformer.generate_dictionary(tof, with_envelope = with_envelope)
H = dformer.get_SAFT_matrix(apf) 
del tof, apf, dformer
print('H = {}S'.format(round(time.time() - start_dict, 3)))

# Reco
if with_envelope == False:
    b = np.dot(H.T, data.flatten('F'))
else:
    env = np.abs(scsi.hilbert(data, axis = 0))
    b = np.abs(np.dot(H.T, env.flatten('F')))
B = np.reshape(b, (M, Nx), 'F')
max_SAFT = np.abs(B).max()

# Plot
plot_reco(1, data, 'B-Scan', dx, dz)
plot_reco(2, B, 'R_track', dx, dz)

del B, b

# %% SSR Reco (deconvolution)
from deconvolution import DataCompression
from deconvolution import deconvolution

# =============================================================================
# # A-Scan selection
# x = int(0.5* (xmax - xmin)) - 8
# M_cp = int(0.2* M)
# 
# # Data compression: time domain -> freq. domain
# dc = DataCompression({'N': M, 'fS': fS, 'fC': fC, 'B': alpha, 'Nt_offset' :  Nt_offset})
# cp = dc.get_compressed_data(data[:, x], M_cp, H[x* M: (x + 1)*M, x* M: (x + 1)*M]) # Compressed data, M_cp x 1
# #del H
# PhiH = dc.get_PhiH() # = np.dot(Phi, H), M_cp x M, normalized
# 
# # Deconvolution
# b_dc, _ = deconvolution(cp, PhiH, 10**-7, True)
# 
# # A-Scan reconstruction
# a_dc = np.dot(H[x* M: (x + 1)*M, (x-5)* M: (x-4)*M], b_dc)
# 
# plt.figure(3)
# plt.plot(data[:, x], label = 'Original')
# #plt.plot(np.abs(scsi.hilbert(data, axis = 0))[:, x], label = 'Envelope')
# plt.plot(a_dc, label = 'SSR Reco')
# plt.title('A-Scan + Envelope')
# plt.legend()
# 
# plt.figure(4)
# plt.plot(b_dc)
# plt.title('Deconvolved data')
# =============================================================================

# %% FISTA Reco
# =============================================================================
# from fista import FISTA
# import scipy.sparse.linalg as sclin
# 
# Niteration = 10
# Lambda = 0.6* max_SAFT
# 
# # SVD for stepsize calculation
# L = sclin.svds(H, tol = 10, maxiter = 50, return_singular_vectors = False)[0]**2
# 
# # Reco
# print('#==========================#')
# print('FISTA!')
# start = time.time()
# 
# fista = FISTA(Lambda, Niteration)
# fista.compute(H, data.flatten('F'), L = L)
# r = fista.get_solution()
# 
# display_time(np.around(time.time() - start, 3))
# plot_reco(5, np.reshape(r, (M, Nx), 'F'), 'Reco FISTA', dx, dz)
# =============================================================================

# %% FISTA using fastmat
import fastmat as fm
import fastmat.algorithms as fma

Lambda = 0.7* max_SAFT

print('FISTA!')
start = time.time()

# Convert the measurement dictionary into fastmat matrix format
matH = fm.Matrix(H)
# FISTA using fastmat
fista = fma.FISTA(matH, numLambda = Lambda, numMaxSteps = 50)
r = fista.process(data.flatten('F'))

display_time(round(time.time() - start, 3))

plot_reco(5, np.reshape(r, (M, Nx), 'F'), 'Reco FISTA', dx, dz)


# %% BEC functions 
def tracking_error_correction(bec, data, p_track, Niteration, epsilon, e_max, triangular = False):
    # Hyperbola fit -> estimate the defect position
    # Point source
    #bec.hyperbola_fit(data, p_track[:, 0])
    # Blurred = representing the physical size
    #bec.hyperbola_fit(data, p_track[:, 0], blur = True, sigma_z = 10, sigma_x = 1)
    # Using SAFT reco
    p_opt = np.copy(p_track)
    for iteration in range(Niteration):
        print('x_opt = {}'.format(p_opt[:, 0]*10**3))
        H_track = bec.dictionary_formation(p_opt, with_envelope = with_envelope)
        bec.defmap_est = np.dot(H_track, data.flatten('F'))
        del H_track
        # Error correction
        x_opt, deltax_opt, _ = bec.error_correction(p_opt, data.flatten('F'),Niteration, epsilon, e_max, triangular)
        # x_opt -> p_opt
        p_opt = np.zeros(p_track.shape)
        p_opt[:, 0] = x_opt
        print('x_opt = {}'.format(p_opt[:, 0]*10**3))
    return p_opt, deltax_opt

def calculate_Reco(bec, ascans, p, with_envelope = False, with_Jacobian = False, deltax = None):
    M = bec.M
    Nx = bec.Nx
    print('Start calculating the dictionary!')
    # Without spatial approximation (for H_true and H_track)
    if with_Jacobian == False: 
        H = bec.dictionary_formation(p, with_envelope = with_envelope)
    # With spatial approximation (for H_opt)
    else:
        H_base, J_base = bec.dictionary_formation(p, with_envelope = with_envelope, with_Jacobian = True)
        H = H_base + np.dot(np.kron(np.diag(deltax), np.identity(M)), J_base)
        del H_base, J_base
    print('End of dictionary calculation')
    reco = np.dot(H.T, ascans)
    Reco = np.reshape(reco, (M, Nx), 'F')
    del H
    return Reco



#======================================================================================================= BEC Reco =====#
# =============================================================================
# fwm_param = {'Nx' : Nx, 'Nz' : Nz, 'Nt' : Nz, 'c0' : c0, 'fS' : fS, 'fC' : fC, 'alpha' : alpha, 'dx' : dx,
#              'opening_angle' : opening_angle}
# 
# bec = BlindErrorCorrection2D(fwm_param, Nt_offset, r)
# 
# data_svd = denoise_svd(data, d = 3)
# 
# start_recoopt = time.time()
# p_opt, deltax_opt = tracking_error_correction(bec, data_svd, p_scan, 10, 0.02, 10* dx, False)
# R_opt = calculate_Reco(bec, data.flatten('F'), p_opt, False, True, deltax_opt)
# # Plots
# plot_reco(4, R_opt, 'R_opt', dx, dz)
# print('R_opt calculation:')
# display_time(time.time() - start_recoopt)
# 
# # defmap
# dm_hat_ps = np.reshape(bec.defmap_est, (data.shape), 'F')
# plot_reco(7, dm_hat_ps, 'Estimated def map, point source', dx, dz)
# =============================================================================


