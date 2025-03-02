# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 10:57:10 2020

@author: sako5821
"""

# -*- coding: utf-8 -*-
"""
#===========================================#
    SmartInspect raw data visualization
#===========================================#

Created on Tue Jun  9 13:24:40 2020

@author: Sayako Kodera
"""
import numpy as np
import matplotlib.pyplot as plt

from ultrasonic_imaging_python.utils.file_readers import SmartInspectReader
from smart_inspect_data_formatter import SmartInspectDataFormatter
from smart_inspect_data_formatter import data_smoothing

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
    
   

def plot_bscan(figure_no, data, title, dx, dz):
    fig = plt.figure(figure_no)
    ax = fig.add_subplot(111)
    ax.imshow(data)
    ax.set_aspect(dz/dx)
    ax.set_title(title)
    ax.set(xlabel = 'x / dx', ylabel = 'z / dz')
    #ax.set_xticks(list(np.linspace(0, data.shape[1] - 1, num = 5)))
    #ax.set_xticklabels([22, 34, 46, 58, 70])


def plot_cscan(figure_no, data, title, dx, dy):
    fig = plt.figure(figure_no)
    ax = fig.add_subplot(111)
    ax.imshow(data)
    ax.set_aspect(dy/dx)
    ax.set_title(title)
    ax.set(xlabel = 'x / dx', ylabel = 'y / dy')

#======================================================================================================================#
date = '200903_2'
path = 'SmartInspect_data/{}/'.format(date)

# binary files
#f = open('{}/XSignal_Time.bin'.format(path), 'rb')
#tim = list(f.read())
#f.close()

# Raw measurement data
si_reader = SmartInspectReader(path)
data_all = si_reader.get_all()
Nt = data_all['data'].shape[0] # Including the back wall echo

# Specify ROI
tmin, tmax = 90, 500 #230, 330 = Nt_offset, Nt
xmin = None#data_all['positions'][:, 0].min() + 1055
xmax = None#data_all['positions'][:, 0].min() + 1070
ymin = data_all['positions'][:, 1].min() + 100
ymax = data_all['positions'][:, 1].min() + 200

# Data matrix
formatter = SmartInspectDataFormatter(data_all['positions'], xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax)
A_pixel = formatter.get_data_matrix_pixelwise(data_all['data'][tmin:tmax, :])
A_pixel = A_pixel/np.abs(A_pixel).max() # Normalize
C_pixel = np.max(np.abs(A_pixel), axis = 0).T

# Data smoothing
#w = np.array([0, 1/6, 0, 1/6, 1/3, 1/3, 0, 1/6, 0]) # 5 pixelaveraging
# =============================================================================
# w = 1/(3 + np.sqrt(2))* np.array([
#         1/(2*np.sqrt(2)), 1/2, 1/(2*np.sqrt(2)),
#         1/2, 1, 1/2,
#         1/(2*np.sqrt(2)), 1/2, 1/(2*np.sqrt(2))])
# =============================================================================
#w = 1/25*np.ones((25, 1))
#A_aa = data_smoothing(A_pixel, 5, 5, w)
#C_aa = np.max(np.abs(A_aa), axis = 0).T

# Plots
c0 = 5920 #[m/S]
fS = 80*10**6 #[Hz] 
dx = (260/1330)*10**-3 #[m]
dy = (130/667)*10**-3 #[m]
dz = 0.5* c0/(fS)
y = 50#331 - ymin

plot_cscan(1, C_pixel, 'C-Scan (raw data)', dx, dy)
#plot_cscan(2, C_aa, 'C-Scan (area averaged)', dx, dy)
#plot_bscan(3, A_pixel[:, :, y], 'B-Scan at y = {} (raw data)'.format(y), dx, dz)
#plot_bscan(4, A_aa[:, :, y], 'B-Scan at y = {} (area averaged)'.format(y), dx, dz)


# Save data
from tools.npy_file_writer import save_data
#save_data(A_pixel, 'npy_data/SmartInspect/{}'.format(date), 'A_pixel.npy')
#save_data(A_aa, 'npy_data/SmartInspect/{}'.format(date), 'A_aa.npy')
sidata = data_all['data'][tmin:tmax, :]
save_data(sidata, 'npy_data/SmartInspect/{}'.format(date), 'raw_data.npy')
save_data(data_all['positions'], 'npy_data/SmartInspect/{}'.format(date), 'p_scan.npy')