#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###### convert 2D array into tex-data ######
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
from matplotlib import cm
import matplotlib.colors as mc

import tools.tex_2D_visualization as pgf2d
import tools.array2image as arr2im
from tools.ColorMap import generate_color_map
from array_access import search_vector


def savepng_batch_itp(batch_3d, path, fpng, vmax):
    # Convert to C-Scan
    cscan = convert2cscan(batch_3d)
    # Convert to image formatting
    cscan_img = get_cscanimg_png(cscan)
    
    ### Convert the array into png ###
    # Image setting
    colors = ['TUI_white', 'TUI_orange_light', 'TUI_orange_dark',  'TUI_red_dark']
    boundaries = np.array([0.00, 0.35, 0.75, 1.00]) #[0.00, 0.43, 0.50, 0.57, 1.00]
    jsonFile = 'tools/TUI_FRI_colors.json'
    # Save
    arr2im.get_image(cscan_img, colors, boundaries, jsonFile, '{}/{}'.format(path, fpng), input_vmin_vmax = True,
                     vmin_input = 0, vmax_input = vmax, normalize = False)
    
    
def savepng_batch_itp_rmse(rmse, path, fpng):
    # Convert to image formatting
    img = get_cscanimg_png(rmse)
    # Get the vmax
    vmax = np.nanmax(rmse)
    print('vmax = {}'.format(vmax))
    # Image setting
    colors = ['TUI_white', 'TUI_orange_light', 'TUI_orange_dark',  'TUI_red_dark']
    boundaries = np.array([0.00, 0.35, 0.75, 1.00]) #[0.00, 0.43, 0.50, 0.57, 1.00]
    jsonFile = 'tools/TUI_FRI_colors.json'
    # Save
    arr2im.get_image(img, colors, boundaries, jsonFile, '{}/{}'.format(path, fpng), input_vmin_vmax = True,
                     vmin_input = 0, vmax_input = vmax, normalize = False)
    
    

def savepng_batch_itp_spectra(spectra, path, fpng, vmax = None):
    # Convert to image formatting
    img = get_cscanimg_png(np.abs(spectra))
    # Get the vmax
    if vmax is None:
        vmax = np.nanmax(np.abs(spectra))
    print('vmax = {}'.format(vmax))
    # Image setting
    colors = ['TUI_white', 'TUI_orange_light', 'TUI_orange_dark',  'TUI_red_dark']
    boundaries = np.array([0.00, 0.35, 0.75, 1.00]) #[0.00, 0.43, 0.50, 0.57, 1.00]
    jsonFile = 'tools/TUI_FRI_colors.json'
    # Save
    arr2im.get_image(img, colors, boundaries, jsonFile, '{}/{}'.format(path, fpng), input_vmin_vmax = True,
                     vmin_input = 0, vmax_input = vmax, normalize = False)

    

def savepng_batch_itp_error(err_3D, path, fpng):
    # Calculate the norm
    err_2D = np.linalg.norm(err_3D, axis = 0)
    # Convert to image formatting
    img = get_cscanimg_png(err_2D)
    # Get the vmax
    vmax = np.nanmax(err_2D)
    print('vmax = {}'.format(vmax))
    # Image setting
    colors = ['TUI_white', 'TUI_orange_light', 'TUI_orange_dark',  'TUI_red_dark']
    boundaries = np.array([0.00, 0.35, 0.75, 1.00]) #[0.00, 0.43, 0.50, 0.57, 1.00]
    jsonFile = 'tools/TUI_FRI_colors.json'
    # Save
    arr2im.get_image(img, colors, boundaries, jsonFile, '{}/{}'.format(path, fpng), input_vmin_vmax = True,
                     vmin_input = 0, vmax_input = vmax, normalize = False)



def savepng(data_3d, path, fpng, normalize = True, log_scale = False, list_vmin_vmax = None):
    #### (1) Convert to the image format
    # Convert to C-Scan
    cscan = convert2cscan(data_3d)
    # Normalize the data
    if normalize == True:
        cscan_max = abs(cscan).max()
        cscan = cscan / cscan_max
    
    # Setting the vmin & vmax
    if list_vmin_vmax is not None:
        vmin, vmax = list_vmin_vmax
    else:
        # For just normalizing
        if log_scale == False:
            vmin, vmax = 0, 1
        # For log scaling
        else:
            vmin = None
            vmax = 0
    
    # Log scaling
    if log_scale == True:
        cscan = log_scaling(cscan, vmin)
        
    # Convert to image formatting
    cscan_img = get_cscanimg_png(cscan)
    
    
    ### (2) Convert the array into png 
    # Image setting
    colors = ['TUI_white', 'orange_very_light',  'TUI_orange_light', 'TUI_orange_dark', 'orange_red',
              'TUI_red_dark', 'FRI_red_dark']
    #boundaries = np.array([0.00, 0.02, 0.12, 0.75, 0.85, 0.97, 1.00]) # 0.00, 0.45, 0.5, 0.55, 0.9, 1.00
    boundaries = np.array([0.00, 0.06, 0.12, 0.75, 0.9, 0.97, 1.00]) # 0.00, 0.45, 0.5, 0.55, 0.9, 1.00
    jsonFile = 'tools/TUI_FRI_colors.json'
    
    # Save
    arr2im.get_image(cscan_img, colors, boundaries, jsonFile, '{}/{}'.format(path, fpng), input_vmin_vmax = True,
                     vmin_input = vmin, vmax_input = vmax, normalize = False)
    # Print color map
    texcmap = pgf2d.TeXcmap(colors, boundaries, jsonFile)
    print('%===== Color map =====%')
    print(texcmap)
    
    
    
def savepng_sideview(data_3d, path, fpng, y_slice, z_range = None):
    # B-Scan
    if z_range is None:
        bscan = data_3d[:, :, y_slice]
    else:
        bscan = data_3d[z_range[0]:z_range[1], :, y_slice]
    
    ### Convert the array into png ###
    # Image setting
    colors = ['FRI_Gray_customized','TUI_blue_dark', 'TUI_blue_light', 'FRI_blue_ultra_light', 'TUI_white', 
              'orange_very_light', 'TUI_orange_light', 'TUI_orange_dark','TUI_red_dark', 'FRI_red_dark']
    #boundaries = np.array([0.00, 0.47, 0.5, 0.53, 0.9, 1.00]) 
    boundaries = np.array([0.00, 0.1, 0.47, 0.49, 0.5, 0.51, 0.53, 0.8, 0.95, 1.00]) #[0.00, 0.43, 0.50, 0.57, 1.00]
    jsonFile = 'tools/TUI_FRI_colors.json'
    # Save
    arr2im.get_image(bscan, colors, boundaries, jsonFile, '{}/{}'.format(path, fpng), input_vmin_vmax = False, 
                     normalize = False)
    # Print color map
    texcmap = pgf2d.TeXcmap(colors, boundaries, jsonFile)
    print('%===== Color map =====%')
    print(texcmap)



def log_scaling(data, vmin):
    """

    Parameters
    ----------
    data : np.ndarray
    vmin : float, NEGATIVE! 
        Minimum value for color map. 
        The values are cut off at this value to make smaller values "white". The default is -40.
        
    Return
    ------
        log_data : np.ndarray
            Log scaled data
    """
    if vmin is None:
        vmin = -27.5
    offset = np.power(10, 2*(vmin-1)/20) #add this to the data to to avoid 0s (other it cause - infty)
    return 20*np.log10(np.abs(data+offset))




def convert2cscan(data_3d, log_scale = False, vmin = None):
    cscan = np.max(np.abs(data_3d), axis = 0) 
    # Log scaling
    if log_scale == True:
        cscan = log_scaling(cscan, vmin)
    
    return cscan


def get_cscanimg_png(cscan):
    """ imsave has different ordering as numpy! For saving, just rotate!
    """
    cscan_img = np.rot90(cscan)
    return cscan_img


def plot_cscan(data, title, dx, dy, vmin = None, vmax = None, cmap = None):
    # !!!!!!! Swap axes to align to the MUSE CAD image !!!!!
    cscan = np.swapaxes(data, 0, 1)
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cscan, vmin = vmin, vmax = vmax, cmap = cmap)#, extent=[xrange[0], xrange[-1], yrange[0], yrange[-1]])
    ax.set_aspect(dy/dx)
    ax.set_title(title)
    ax.set(xlabel = 'x / dx', ylabel = 'y / dy')
    ax.invert_yaxis()
    plt.colorbar(im)
    
    del fig
    
    
    
def plot_bscan(data, title, dx, dz, vmin = None, vmax = None):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(data, vmin = vmin, vmax = vmax)#, extent=[xrange[0], xrange[-1], yrange[0], yrange[-1]])
    ax.set_aspect(dz/dx)
    ax.set_title(title)
    ax.set(xlabel = 'x / dx', ylabel = 'z / dz')
    ax.invert_yaxis()
    plt.colorbar(im)
    
    del fig


def plot_scan_positions(p_smp, p_resmp, title, dx = 1, dy = 1):
    # !!!!!!! Swap axes to align to the MUSE CAD image !!!!!
    p_smp2plot = np.copy(p_smp)#np.swapaxes(p_smp, 0, 1)
    p_resmp2plot = np.copy(p_resmp)#np.swapaxes(p_resmp, 0, 1)
    
    plt.figure()
    plt.scatter(p_smp2plot[:, 0], p_smp2plot[:, 1], label = 'p_smp')
    plt.scatter(p_resmp2plot[:, 0], p_resmp2plot[:, 1], label = 'p_resmp')
    plt.title(title)
    plt.xlabel('x/dx')
    plt.ylabel('y/dy')


def plot_reco_set(path, setNo):
    # Load raw data
    muse_data = scio.loadmat('MUSE/measurement_data.mat')
    # Select the ROI
    zmin, zmax = 1888, 2400 # = Nt_offset, Nt, 1865....1895 = only noise, 1888, 2400
    xmin, xmax = 185, 235#base = 240, 350   115, 145 
    ymin, ymax = 20, 70 #base = 115, 165
    A_roi = muse_data['data'][zmin:zmax, xmin:xmax, ymin:ymax]
    A_roi = A_roi / np.abs(A_roi).max() # normalize
    
    # Load recos
    res = load_results(path, setNo)
    R_true = load_reco('npy_data/simulations/R_true.npy')#[:, 5:-5, :]
    
    
    ### Plots ###
    # Raw data
    plot_cscan(convert2cscan(A_roi), 'C-Scan of ROI', 1, 1)
    
    # Scan positions: initial points vs random resampling
    plot_scan_positions(res['p_smp'], res['p_resmp_rnd'], 'Sampling vs random resampling positions')
    # Scan positions: initial points vs "optimal" resampling
    plot_scan_positions(res['p_smp'], res['p_resmp_krig'], 'Sampling vs "optimal" resampling positions')
    
    # Recos
    plot_cscan(convert2cscan(R_true), 'R_true', 1, 1, vmin = None, vmax = None)
    plot_cscan(convert2cscan(res['R_smp']), 'R_smp', 1, 1, vmin = None, vmax = None)
    plot_cscan(convert2cscan(res['R_idw']), 'R_idw', 1, 1, vmin = None, vmax = None)
    plot_cscan(convert2cscan(res['R_re_idw']), 'R_re_idw', 1, 1, vmin = None, vmax = None)
    plot_cscan(convert2cscan(res['R_fk']), 'R_fk', 1, 1, vmin = None, vmax = None)
    plot_cscan(convert2cscan(res['R_re_fk']), 'R_re_fk', 1, 1, vmin = None, vmax = None)
    
    
def plot_reco_set_logscale(path, setNo, vmin = -27.5, vmax = 0):
    # Load raw data
    muse_data = scio.loadmat('MUSE/measurement_data.mat')
    # Select the ROI
    zmin, zmax = 1888, 2400 # = Nt_offset, Nt, 1865....1895 = only noise, 1888, 2400
    xmin, xmax = 185, 235#base = 240, 350   115, 145 
    ymin, ymax = 20, 70 #base = 115, 165
    A_roi = muse_data['data'][zmin:zmax, xmin:xmax, ymin:ymax]
    A_roi = A_roi / np.abs(A_roi).max() # normalize
    
    # Load recos
    res = load_results(path, setNo)
    R_true = load_reco('npy_data/simulations/R_true.npy')
    
    
    ### Plots ###
    # Cutomize color map
    # custom = np.zeros((257,4))
    # custom[1:257,:] = cm.get_cmap('Oranges', 256)(np.arange(0,256))
    # custom = mc.ListedColormap(custom)
    
    # Customized color map
    # Image setting
    colors = ['TUI_white', 'orange_very_light',  'TUI_orange_light', 'TUI_orange_dark', 'orange_red',
              'TUI_red_dark', 'FRI_red_dark']
    #boundaries = np.array([0.00, 0.02, 0.12, 0.75, 0.85, 0.97, 1.00]) # 0.00, 0.45, 0.5, 0.55, 0.9, 1.00
    boundaries = np.array([0.00, 0.06, 0.12, 0.75, 0.9, 0.97, 1.00]) # 0.00, 0.45, 0.5, 0.55, 0.9, 1.00
    jsonFile = 'tools/TUI_FRI_colors.json'
    # Generate
    custom = generate_color_map(colors, boundaries, jsonFile)
    
    
    # Raw data
    plot_cscan(convert2cscan(A_roi, log_scale = True, vmin = vmin), 'C-Scan of ROI', 1, 1,
               vmin = vmin, vmax = vmax, cmap = custom)
    
    # Scan positions: initial points vs random resampling
    plot_scan_positions(res['p_smp'], res['p_resmp_rnd'], 'Sampling vs random resampling positions')
    # Scan positions: initial points vs "optimal" resampling
    plot_scan_positions(res['p_smp'], res['p_resmp_krig'], 'Sampling vs "optimal" resampling positions')
    
    # Recos: C-scans are log scaled!!!
    plot_cscan(convert2cscan(R_true, log_scale = True, vmin = vmin), 'R_true', 1, 1, 
               vmin = vmin, vmax = vmax, cmap = custom)
    plot_cscan(convert2cscan(res['R_smp'], log_scale = True, vmin = vmin), 'R_smp', 1, 1, 
               vmin = vmin, vmax = vmax, cmap = custom)
    plot_cscan(convert2cscan(res['R_idw'], log_scale = True, vmin = vmin), 'R_idw', 1, 1, 
               vmin = vmin, vmax = vmax, cmap = custom)
    plot_cscan(convert2cscan(res['R_re_idw'], log_scale = True, vmin = vmin), 'R_re_idw', 1, 1, 
               vmin = vmin, vmax = vmax, cmap = custom)
    plot_cscan(convert2cscan(res['R_fk'], log_scale = True, vmin = vmin), 'R_fk', 1, 1, 
               vmin = vmin, vmax = vmax, cmap = custom)
    plot_cscan(convert2cscan(res['R_re_fk'], log_scale = True, vmin = vmin), 'R_re_fk', 1, 1, 
               vmin = vmin, vmax = vmax, cmap = custom)    



def load_reco(fname):
    reco = np.load(fname)
    return reco / np.abs(reco).max()

    
    
def load_results(path, setNo):
    data2load = ['p_smp', 'p_resmp_rnd', 'p_resmp_krig', 'R_smp', 'R_idw', 'R_re_idw', 'R_fk', 'R_re_fk',
                 'rmse_FK', 'rmse_re_FK']
    
    results = {}
    
    for item in data2load:
        data = np.load('{}/setNo_{}/{}.npy'.format(path, int(setNo), item))
        # Updaate the dictionary
        results.update({
            item : data
            })
    
    return results


def unique_resampling_positions(p_smp, p_resmp):
    # Find the unique resampling positions
    p_resmp_all = np.concatenate((p_smp, p_resmp))
    rows = []
    notunique = []
    for row, p in enumerate(p_resmp_all):
        indices = search_vector(p, p_resmp_all, axis = 1)
        if len(indices) > 1:
            rows.append(row)
            notunique.append(indices)
    p_resmp_all_unique = np.delete(p_resmp_all, rows[:int(len(rows)/2)], axis = 0)
    print(p_resmp_all_unique.shape)
    return p_resmp_all_unique
    
    



def save_recopng_set(path_npy, setNo, coverage_str, with_trues = False):
    # TeX setup
    path_png_base = '/Users/sayakokodera/Uni/Master/MA/tex/MA/figures/tex_png/simulations'
    path_png = '{}/{}'.format(path_png_base, coverage_str)
    
    # Load recos
    res = load_results(path_npy, setNo)
    
    # Save recos
    savepng(res['R_smp'], path_png, 'R_smp.png')
    savepng(res['R_idw'], path_png, 'R_idw.png')
    savepng(res['R_re_idw'], path_png, 'R_re_idw.png')
    savepng(res['R_fk'], path_png, 'R_fk.png')
    savepng(res['R_re_fk'], path_png, 'R_re_fk.png')
    
    
    # A_smp
    from smart_inspect_data_formatter import SmartInspectDataFormatter
    # Load true data
    # Select the ROI
    zmin, zmax = 1888, 2400 # = Nt_offset, Nt, 1865....1895 = only noise, 1888, 2400
    xmin, xmax = 185, 235#base = 240, 350   115, 145 
    ymin, ymax = 20, 70 #base = 115, 165
    A_true = np.load('npy_data/simulations/A_true.npy')
    # Format A_smp
    p_smp = res['p_smp']
    formatter = SmartInspectDataFormatter(p_smp, xmin = 0, xmax = (xmax - xmin), 
                                          ymin = 0, ymax = (ymax - ymin))
    A_smp = formatter.get_data_matrix_pixelwise(A_true[:, p_smp[:, 0], p_smp[:, 1]])
    # Save
    savepng(A_smp, path_png, 'A_smp.png')
    
    # Format A_resmp
    # Find the unique resampling positions
    p_resmp_all_unique = unique_resampling_positions(p_smp, res['p_resmp_krig'])
    # Format
    formatter = SmartInspectDataFormatter(p_resmp_all_unique, xmin = 0, xmax = (xmax - xmin), 
                                          ymin = 0, ymax = (ymax - ymin))
    A_resmp = formatter.get_data_matrix_pixelwise(A_true[:, p_resmp_all_unique[:, 0], p_resmp_all_unique[:, 1]])
    # Save
    savepng(A_resmp, path_png, 'A_resmp.png')
    
    
    # Save true data and recos
    if with_trues == True:
        savepng(A_true, path_png_base, 'A_true.png')
        # R_true
        R_true = np.load('npy_data/simulations/R_true.npy')
        savepng(R_true, path_png_base, 'R_true.png')
        
        
def save_recopng_logscaled_set(path_npy, setNo, coverage_str, with_trues = False):
    # TeX setup
    path_png_base = '/Users/sayakokodera/Uni/Master/MA/tex/Defense/figures/tex_png/simulations_logscale'
    path_png = '{}/{}'.format(path_png_base, coverage_str)
    
    # Load recos
    res = load_results(path_npy, setNo)
    
    # Save recos
    savepng(res['R_smp'], path_png, 'R_smp.png', log_scale = True, list_vmin_vmax = [-27.5, 0])
    savepng(res['R_idw'], path_png, 'R_idw.png', log_scale = True, list_vmin_vmax = [-27.5, 0])
    savepng(res['R_re_idw'], path_png, 'R_re_idw.png', log_scale = True, list_vmin_vmax = [-27.5, 0])
    savepng(res['R_fk'], path_png, 'R_fk.png', log_scale = True, list_vmin_vmax = [-27.5, 0])
    savepng(res['R_re_fk'], path_png, 'R_re_fk.png', log_scale = True, list_vmin_vmax = [-27.5, 0])
    
    
    # A_smp
    from smart_inspect_data_formatter import SmartInspectDataFormatter
    # Load true data
    # Select the ROI
    zmin, zmax = 1888, 2400 # = Nt_offset, Nt, 1865....1895 = only noise, 1888, 2400
    xmin, xmax = 185, 235#base = 240, 350   115, 145 
    ymin, ymax = 20, 70 #base = 115, 165
    A_true = np.load('npy_data/simulations/A_true.npy')
    # Format A_smp
    p_smp = res['p_smp']
    formatter = SmartInspectDataFormatter(p_smp, xmin = 0, xmax = (xmax - xmin), 
                                          ymin = 0, ymax = (ymax - ymin))
    A_smp = formatter.get_data_matrix_pixelwise(A_true[:, p_smp[:, 0], p_smp[:, 1]])
    # Save
    savepng(A_smp, path_png, 'A_smp.png', log_scale = True, list_vmin_vmax = [-27.5, 0])
    
    # Format A_resmp
    # Find the unique resampling positions
    p_resmp_all_unique = unique_resampling_positions(p_smp, res['p_resmp_krig'])
    # Format
    formatter = SmartInspectDataFormatter(p_resmp_all_unique, xmin = 0, xmax = (xmax - xmin), 
                                          ymin = 0, ymax = (ymax - ymin))
    A_resmp = formatter.get_data_matrix_pixelwise(A_true[:, p_resmp_all_unique[:, 0], p_resmp_all_unique[:, 1]])
    # Save
    savepng(A_resmp, path_png, 'A_resmp.png', log_scale = True, list_vmin_vmax = [-27.5, 0])

    
    # Save true data and recos
    if with_trues == True:
        savepng(A_true, path_png_base, 'A_true.png', log_scale = True, list_vmin_vmax = [-27.5, 0])
        # R_true
        R_true = np.load('npy_data/simulations/R_true.npy')
        savepng(R_true, path_png_base, 'R_true.png', log_scale = True, list_vmin_vmax = [-27.5, 0])        

    
    
def save_recopng_set_sideview(path_npy, setNo, coverage_str, y_slice, z_range = None, with_trues = False):
    """
    Parameters
    ----------
        z_range: list [zimg_min, zimg_max]
            Image range for z-axis
    """
    # TeX setup
    path_png_base = '/Users/sayakokodera/Uni/Master/MA/tex/MA/figures/tex_png/simulations'
    path_png = '{}/{}'.format(path_png_base, coverage_str)
    
    # Load recos
    res = load_results(path_npy, setNo)
    
    # Save recos
    savepng_sideview(res['R_smp'], path_png, 'R_smp_sideview.png', y_slice, z_range)
    savepng_sideview(res['R_idw'], path_png, 'R_idw_sideview.png', y_slice, z_range)
    savepng_sideview(res['R_re_idw'], path_png, 'R_re_idw_sideview.png', y_slice, z_range)
    savepng_sideview(res['R_fk'], path_png, 'R_fk_sideview.png', y_slice, z_range)
    savepng_sideview(res['R_re_fk'], path_png, 'R_re_fk_sideview.png', y_slice, z_range)
    
    
    # A_smp
    from smart_inspect_data_formatter import SmartInspectDataFormatter
    # Load true data
    # Select the ROI
    zmin, zmax = 1888, 2400 # = Nt_offset, Nt, 1865....1895 = only noise, 1888, 2400
    xmin, xmax = 185, 235#base = 240, 350   115, 145 
    ymin, ymax = 20, 70 #base = 115, 165
    A_true = np.load('npy_data/simulations/A_true.npy')
    # Format A_smp
    p_smp = res['p_smp']
    formatter = SmartInspectDataFormatter(p_smp, xmin = 0, xmax = (xmax - xmin), 
                                          ymin = 0, ymax = (ymax - ymin))
    A_smp = formatter.get_data_matrix_pixelwise(A_true[:, p_smp[:, 0], p_smp[:, 1]])
    # Save
    savepng_sideview(A_smp, path_png, 'A_smp_sideview.png', y_slice, z_range)
    
    # Save true data and recos
    if with_trues == True:
        savepng_sideview(A_true, path_png_base, 'A_true_sideview.png', y_slice, z_range)
        # R_true
        R_true = np.load('npy_data/simulations/R_true.npy')
        savepng_sideview(R_true, path_png_base, 'R_true_sideview.png', y_slice, z_range)
    

def savepng_simulations_rmse(rmse, path, fpng):
    # Convert to image formatting
    img = get_cscanimg_png(rmse)
    # manipulation
    img[40:42, 40:42] = 100
    # Get the vmax
    vmax = np.nanmax(rmse)
    mean = np.nanmean(rmse)
    print('vmax = {}, mean = {}'.format(vmax, mean))
    # Image setting
    colors = ['TUI_white', 'TUI_orange_dark', 'TUI_red_dark', 'FRI_red_dark']
    boundaries = np.array([0.00, 0.05, 0.25, 1.00]) #[0.00, 0.43, 0.50, 0.57, 1.00]
    jsonFile = 'tools/TUI_FRI_colors.json'
    # Save
    arr2im.get_image(img, colors, boundaries, jsonFile, '{}/{}'.format(path, fpng), input_vmin_vmax = True,
                     vmin_input = mean, vmax_input = vmax, normalize = False)
    # Print color map
    texcmap = pgf2d.TeXcmap(colors, boundaries, jsonFile)
    print('%===== Color map =====%')
    print(texcmap)

