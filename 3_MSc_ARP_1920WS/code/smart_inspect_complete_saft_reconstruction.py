###################### run test script for data synthesizers for manual scan ###########################

import numpy as np
import matplotlib.pyplot as plt
import json

# =============================================================================
# from ultrasonic_imaging_python.forward_models import data_synthesizers_manual_scan
# from ultrasonic_imaging_python.reconstruction.saft_for_manual_scan import SAFTonGridforManualScan
# from ultrasonic_imaging_python.visualization.slice_figures import SliceFigure3D
# from ultrasonic_imaging_python.visualization import c_images_arbitrarily_sampled
# from ultrasonic_imaging_python.definitions import units
# ureg = units.ureg
# from ultrasonic_imaging_python.manual_scans.scan_position_synthesizers import ScanPositionSynthesizer
# =============================================================================

"""
Reconstructing SI datasets of the entire steel object

"""

plt.close('all')

# %% Functions
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


def display_time(stop):
    if stop > 60 and stop < 60**2:
        print('** Time to complete : {} min'.format(round(stop/60, 2)))
    elif stop >= 60**2:
        print('** Time to complete : {} h'.format(round(stop/(60**2), 2)))
    else:
        print('** Time to complete : {} s'.format(round(stop, 2)))


def plot_reco_cscan(figure_no, cscan, title, dx, dy):
    fig = plt.figure(figure_no)
    ax = fig.add_subplot(111)
    ax.imshow(cscan)
    ax.set_aspect(dy/dx)
    ax.set_title(title)
    ax.set(xlabel = 'x / dx', ylabel = 'y / dy')

# %% Measurement data
fdate = '200903_1'
path = 'npy_data/SmartInspect/{}'.format(fdate)
p_scan = np.load('{}/p_scan.npy'.format(path))
data = np.load('{}/raw_data.npy'.format(path))
data_vec = data.flatten('F')

# For specifying the ROI
data_3D = np.load('{}/A_pixel.npy'.format(path))

# =============================================================================
# # Parameters
# fC = 4.48* 10**6 #[Hz], carrier frequncy of the pulse
# D = 35*10**-3 #[m], transducer diameter
# 
# specimen_params = {
#         'dxdata': 260/1892 * ureg.millimeter,
#         'dydata': 130/945 * ureg.millimeter,
#         'c0': 5920 * ureg.meter / ureg.second,#5920
#         'fS': 80 * ureg.megahertz,
#         'Nxdata': data_3D.shape[1],
#         'Nydata': data_3D.shape[2],
#         'Ntdata': data_3D.shape[0],
#         'Nxreco': data_3D.shape[1],
#         'Nyreco': data_3D.shape[2],
#         'Nzreco': data_3D.shape[0],
#         'openingangle': get_opening_angle(5920, D, fC), # degree
#         'anglex': 0, # degree
#         'angley': 0, # degree
#         'zImage': 0 * ureg.meter,
#         'xImage': 0 * ureg.meter,
#         'yImage': 0 * ureg.meter,
#         't0': 90 / (80*10**6) * ureg.second #0
#         }
# 
# del data_3D
# =============================================================================
# %% Reconstruction
# =============================================================================
# # SAFT class
# compute_saft = SAFTonGridforManualScan('3DSingleMedium', enable_file_IO = False)
# compute_saft.set_forward_model_parameters(specimen_params)
# 
# # Specify the measurement positions
# x_scan_mm = p_scan[:, 0]* specimen_params['dxdata']
# y_scan_mm = p_scan[:, 1]* specimen_params['dydata']
# 
# #(9)
# reco = compute_saft.get_reco(data_vec, x_scan_mm, y_scan_mm, save_data = False)
# 
# 
# #%% visualization
# # (10) C-scan visualization of reco
# reco_cimg = compute_saft.generate_cscan_format(reco, summation = True, save_data = False)
# plot_reco_cscan(1, reco_cimg, 'SAFT reco of {} (C-Scan)'.format(fdate), specimen_params['dxdata'].magnitude, 
#                 specimen_params['dydata'].magnitude)
# 
# =============================================================================
