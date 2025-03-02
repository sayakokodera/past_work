# -*- coding: utf-8 -*-
"""
Visualize B-Scan and Reco with a larger ROI
"""
import numpy as np
import matplotlib.pyplot as plt

import tools.array2image as arr2im

plt.close('all')

def plot_data(figure_no, data, title, dx, dz, fname):
    # Image setting
    colors = ['TUI_blue_dark', 'TUI_blue_light', 'TUI_white', 'TUI_orange_light', 'TUI_orange_dark']
    boundaries = np.array([0.00, 0.43, 0.50, 0.57, 1.00])
    jsonFile = 'tools/TUI_FRI_colors.json'
    # Normalize the data
    data_max = abs(data).max()
    data = data / data_max
    # Convert the array into png
    arr2im.get_image(data, colors, boundaries, jsonFile, fname, input_vmin_vmax = False)
    # Plot
    fig = plt.figure(figure_no)
    ax = fig.add_subplot(111)
    ax.imshow(data)
    ax.set_aspect(dz/dx)
    ax.set_title(title)
    ax.set(xlabel = 'x [mm]', ylabel = 'z [mm]')
    #ax.set_xlim([15, 25])
    ax.set_xticks([0, 15, 23, 25, 27, 35, 49])#
    ax.set_xticklabels([7.5, 15, r'$-\lambda$', 20, r'$\lambda$', 25, 32.5])
    ax.set_yticks([0, 68, 100, 132, 199])
    ax.set_yticklabels([25, r'$-\lambda$', 30, r'$\lambda$', 35])


# Lodd data
Bscan = np.load('npy_data/200401/30mm/Bscan_largeROI.npy')
Rtrue = np.load('npy_data/200401/30mm/Reco_true_largeROI.npy')
# Normalize the data
Bscan = Bscan / np.abs(Bscan).max()
Rtrue = Rtrue / np.abs(Rtrue).max()



dx = 0.5*10**-3 #[m]
dz = 3.9375e-05 #[m]
path = 'tex_pngs/200401'
plot_data(1, Bscan, 'B-Scan', dx, dz, '{}/30mm_Bscan.png'.format(path))
plot_data(2, Rtrue, 'Reconstruction', dx, dz, '{}/30mm_true_largeROI.png'.format(path))