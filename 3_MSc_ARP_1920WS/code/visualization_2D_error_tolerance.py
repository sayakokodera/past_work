# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 22:43:16 2020

@author: johan
"""

import numpy as np
import matplotlib.pyplot as plt

import tools.array2image as arr2im

def savepng(data, Nx_roi, Nz_roi, datatype, zdef, err):
    # Image setting
    colors = ['TUI_blue_dark', 'TUI_blue_light', 'TUI_white', 'TUI_orange_light', 'TUI_orange_dark']
    boundaries = np.array([0.00, 0.43, 0.50, 0.57, 1.00])
    jsonFile = 'tools/TUI_FRI_colors.json'
    # Normalize the data
    data_max = abs(data).max()
    data = data / data_max
    # Select the image range of the array
    xmin = int(Nx_roi/2) - int(data.shape[1]/2)
    xmax = xmin + data.shape[1]
    zmin = int((Nz_roi - data.shape[0])/2)
    zmax = zmin + data.shape[0]
    data_img = np.zeros((Nz_roi, Nx_roi))
    data_img[zmin:zmax, xmin:xmax] = np.copy(data)
    fpng = 'tex_pngs/200401/{}mm_{}_{}lambda.png'.format(zdef, datatype, err)
    # Convert the array into png
    arr2im.get_image(data_img, colors, boundaries, jsonFile, fpng, input_vmin_vmax = False)

plt.close('all')

err_all = [0.4]
zdef = 30
path = 'npy_data/200401/{}mm'.format(zdef)

# Reference: Reco_true
true = np.load('{}/Reco_true.npy'.format(path))
idx = 0# 25, 107, 47, 69, 84, 133, 164

dx = 0.5 #[mm]
dz = 3.9375*10**-2 #[mm]
Nx_roi = 21
Nz_roi = round(dx*(Nx_roi - 1)/dz)

for err in err_all:
    # Load data
    track = np.load('{}/{}lambda/Reco_track/No.{}.npy'.format(path, err, idx)) 
    opt = np.load('{}/{}lambda/Reco_opt/No.{}.npy'.format(path, err, idx))
    # plots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.suptitle('Error {}lambda'.format(err))
    ax1.imshow(true)
    ax1.set_title('true')
    ax2.imshow(track)
    ax2.set_title('track')
    ax3.imshow(opt)
    ax3.set_title('opt')
    # Save pngs
    #savepng(track, Nx_roi, Nz_roi, 'track', zdef, err)
    #savepng(opt, Nx_roi, Nz_roi, 'opt', zdef, err)
    
