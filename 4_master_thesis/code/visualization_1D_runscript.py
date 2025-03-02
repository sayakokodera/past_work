#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###### convert 2D array into tex-data ######
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio

import tools.tex_1D_visualization as pgf1d
from tools.npy_file_writer import num2string


def plot_fv_comp_singlebin(fbin):
    lags = np.load('npy_data/batch_itp/lags_full.npy')
    fv_full = np.load('npy_data/batch_itp/fv_full.npy')
    fv_smp = np.load('npy_data/batch_itp/fv_smp.npy')
    fv_DNN = np.load('npy_data/batch_itp/fv_DNN.npy')
    
    # plot
    plt.figure()
    plt.plot(lags, fv_full[fbin, :], label = 'full')
    plt.plot(lags, fv_smp[fbin, :], label = 'smp')
    plt.plot(lags, fv_DNN[fbin, :], label = 'DNN')
    plt.title('FV comp. fbin = {}'.format(fbin))
    plt.legend()  
    

def plot_fv_full_singlebin(fbin):
    lags = np.load('npy_data/batch_itp/lags_full.npy')
    lags = np.around(lags / (0.5* 10** -3), 9)
    fv_full = np.load('npy_data/batch_itp/fv_full.npy')
    
    # plot
    plt.figure()
    plt.plot(lags, fv_full[fbin, :], label = '{}'.format(fbin))
    plt.title('FV (full) fbin = {}'.format(fbin))
    plt.xlabel('lags / dx')
    plt.ylabel('Inc. variances')
    
def plot_fv_DNN_singlebin(fbin):
    lags = np.load('npy_data/batch_itp/lags_full.npy')
    lags = np.around(lags / (0.5* 10** -3), 9)
    fv = np.load('npy_data/batch_itp/fv_DNN.npy')
    
    # plot
    plt.figure()
    plt.plot(lags, fv[fbin, :], 'tab:green', label = '{}'.format(fbin))
    plt.title('FV (DNN FK) fbin = {}'.format(fbin))
    plt.xlabel('lags / dx')
    plt.ylabel('Inc. variances')
    
def plot_fv_smp_singlebin(fbin):
    lags = np.load('npy_data/batch_itp/lags_full.npy')
    lags = np.around(lags / (0.5* 10** -3), 9)
    fv = np.load('npy_data/batch_itp/fv_smp.npy')
    
    # plot
    plt.figure()
    plt.plot(lags, fv[fbin, :], 'tab:orange', label = '{}'.format(fbin))
    plt.title('FV (subsampled) fbin = {}'.format(fbin))
    plt.xlabel('lags / dx')
    plt.ylabel('Inc. variances')


def save_batchitp_ascans(p):
    # Unpack the position
    x, y = p
    # Load a single A-Scan from all data types
    path_npy = 'npy_data/batch_itp'
    a_true = np.load('{}/A_true.npy'.format(path_npy))[:, x, y] # shape = 512
    a_idw = np.load('{}/A_idw.npy'.format(path_npy))[:, x, y]
    a_fk = np.load('{}/A_fk.npy'.format(path_npy))[:, x, y]
    a_fk_full = np.load('{}/A_fk_full.npy'.format(path_npy))[:, x, y]
    
    # TeX data setup
    dt = 1/(80*10**6)
    zmin, zmax = 1888, 2400
    x_val = np.arange(zmin, zmax)*dt*10**6 # in [micro sec]
    path_tex = '/Users/sayakokodera/Uni/Master/MA/tex/MA/figures/coords_1D/batch_itp'
    
    # Save
    # a_true vs a_idw
    pgf1d.generate_coordinates_for_addplot(x_val, '{}/a_idw_x{}y{}.tex'.format(path_tex, x, y), 
                                           False, ['tui_blue', 'tui_orange'], ['very thick', 'very thick'], 
                                           a_true, a_idw)
    # a_true vs a_fk
    pgf1d.generate_coordinates_for_addplot(x_val, '{}/a_fk_x{}y{}.tex'.format(path_tex, x, y), 
                                           False, ['tui_blue', 'tui_orange'], ['very thick', 'very thick'], 
                                           a_true, a_fk)
    # a_true vs a_fk_full
    pgf1d.generate_coordinates_for_addplot(x_val, '{}/a_fk_full_x{}y{}.tex'.format(path_tex, x, y), 
                                           False, ['tui_blue', 'tui_orange'], ['very thick', 'very thick'], 
                                           a_true, a_fk_full)


def save_fv_singlebin(fbin, colors, fvtype):
    #!!!!! len(f_range) == 4 !!!!!!!
    # Load the data
    lags_full = np.load('npy_data/batch_itp/lags_full.npy') 
    # Load the FV according to the type
    if fvtype in ['full', 'smp', 'DNN']:
        fv = np.load('npy_data/batch_itp/fv_{}.npy'.format(fvtype))
    else:
        raise AttributeError('save_fv_singlebin: fvtype is not known!')
        
    # Setup
    dx = 0.5* 10**-3 #[m]
    x_val = np.around(lags_full / dx, 10)
    path_tex = '/Users/sayakokodera/Uni/Master/MA/tex/Talk_AI_210914/figures/coords_1D/batch_itp/'
    
    # Save -> unroll the FVs
    # Without normalization
    pgf1d.generate_coordinates_for_addplot(x_val, '{}/fv_{}_f{}.tex'.format(path_tex, fvtype, fbin), 
                                           False, colors, ['very thick', 'very thick', 'very thick', 'very thick'], 
                                           fv[fbin, :])


def save_fv_frange(f_range, colors, fvtype):
    #!!!!! len(f_range) == 4 !!!!!!!
    # Load the data
    lags_full = np.load('npy_data/batch_itp/lags_full.npy') 
    fv_full = np.load('npy_data/batch_itp/fv_full.npy')
    # Normalize
    vmax = np.max(fv_full, axis = 1)
    fvnorm_full = fv_full/vmax[:, np.newaxis]
    print(vmax.shape)
    # Setup
    dx = 0.5* 10**-3 #[m]
    x_val = lags_full / dx
    path_tex = '/Users/sayakokodera/Uni/Master/MA/tex/MA/figures/coords_1D/batch_itp/'
    
    # Save -> unroll the FVs
    # Without normalization
    pgf1d.generate_coordinates_for_addplot(x_val, '{}/fv_{}.tex'.format(path_tex, fvtype), 
                                           False, colors, ['very thick', 'very thick', 'very thick', 'very thick'], 
                                           fv_full[f_range[0], :], fv_full[f_range[1], :], fv_full[f_range[2], :], 
                                           fv_full[f_range[3], :])
    # Normalized FV
    pgf1d.generate_coordinates_for_addplot(x_val, '{}/fvnorm_{}.tex'.format(path_tex, fvtype), 
                                           False, colors, ['very thick', 'very thick', 'very thick', 'very thick'], 
                                           fvnorm_full[f_range[0], :], fvnorm_full[f_range[1], :], 
                                           fvnorm_full[f_range[2], :], fvnorm_full[f_range[3], :])
    

def save_fv_comp(fbin):
    # Load the data
    lags_full = np.load('npy_data/batch_itp/lags_full.npy') 
    fv_full = np.load('npy_data/batch_itp/fv_full.npy')
    fv_smp = np.load('npy_data/batch_itp/fv_smp.npy')
    fv_dnn = np.load('npy_data/batch_itp/fv_DNN.npy')
    
    # TeX setup
    dx = 0.5* 10**-3 #[m]
    x_val = lags_full / dx
    path_tex = '/Users/sayakokodera/Uni/Master/MA/tex/MA/figures/coords_1D/batch_itp/'
    linestyle = ['very thick', 'very thick']
    
    # Save 
    # fv_full vs fv_smp
    pgf1d.generate_coordinates_for_addplot(x_val, '{}/fvcomp_smp_f{}.tex'.format(path_tex, fbin), 
                                           False, ['tui_blue', 'fri_green'], linestyle, 
                                           fv_full[fbin, :], fv_smp[fbin, :])
    # fv_full vs fv_dnn
    pgf1d.generate_coordinates_for_addplot(x_val, '{}/fvcomp_dnn_f{}.tex'.format(path_tex, fbin), 
                                           False, ['tui_blue', 'tui_orange'], linestyle,
                                           fv_full[fbin, :], fv_dnn[fbin, :])
    

def save_DNN_results(setNo):
    # Load the data
    x_val = np.load('npy_data/ML/results/lags_normed.npy') 
    y_true = np.load('npy_data/ML/results/setNo_{}/y_true.npy'.format(setNo)) 
    y_pred = np.load('npy_data/ML/results/setNo_{}/y_pred.npy'.format(setNo)) 
    x_input = np.load('npy_data/ML/results/setNo_{}/x_input.npy'.format(setNo)) 
    
    # TeX setup
    path_tex = '/Users/sayakokodera/Uni/Master/MA/tex/MA/figures/coords_1D/ML'
    linestyle = ['very thick', 'very thick', 'very thick']
    
    # Save 
    pgf1d.generate_coordinates_for_addplot(x_val, '{}/setNo_{}.tex'.format(path_tex, setNo), 
                                           False, ['fri_green', 'tui_blue', 'tui_orange'], linestyle, 
                                           x_input, y_true, y_pred)



def save_simulation_results():
    # TeX setup
    x_val = np.array([5, 6, 7, 8, 9, 10, 12, 14, 15, 17]) # in %
    path_tex = '/Users/sayakokodera/Uni/Master/MA/tex/MA/figures/coords_1D/simulations'
    colors = ['tui_blue', 'fri_green', 'tui_orange', 'fri_gray_dark', 'tui_red']
    linestyle = ['very thick', 'very thick', 'very thick', 'very thick', 'very thick']
    
    # Load the data
    dtypes = ['R_smp', 'R_idw', 'R_fk', 'R_re_idw', 'R_re_fk']
    gcnr = np.zeros((len(x_val), len(dtypes)))
    gcnr_true = np.load('npy_data/simulations/evaluation/gcnr/R_true.npy') 
    mse = np.zeros(gcnr.shape)
    
    for idx, item in enumerate(dtypes):
        curr_gcnr = np.load('npy_data/simulations/evaluation/gcnr/{}.npy'.format(item))
        curr_mse = np.load('npy_data/simulations/evaluation/mse/{}.npy'.format(item))
        
        gcnr[:, idx] = curr_gcnr/gcnr_true
        mse[:, idx] = curr_mse
    
    # Save
    # gcnr
    pgf1d.generate_coordinates_for_addplot(x_val, '{}/gcnr.tex'.format(path_tex), 
                                           False, colors, linestyle,
                                           gcnr[:, 0], gcnr[:, 1], gcnr[:, 2], gcnr[:, 3],gcnr[:, 4])
    # gcnr
    pgf1d.generate_coordinates_for_addplot(x_val, '{}/mse.tex'.format(path_tex), 
                                           False, colors, linestyle,
                                           mse[:, 0], mse[:, 1], mse[:, 2], mse[:, 3],mse[:, 4])
                                           

def save_simulation_results_for_defense():
    # TeX setup
    x_val = np.array([5, 6, 7, 8, 9, 10, 12, 14, 15, 17]) # in %
    path_tex = '/Users/sayakokodera/Uni/Master/MA/tex/Defense/figures/coords_1D/simulations/'
    colors = ['tui_blue', 'fri_green', 'tui_orange', 'fri_gray_dark', 'tui_red']
    linestyle = ['very thick', 'very thick', 'very thick', 'very thick', 'very thick']
    
    # Load the data
    dtypes = ['R_smp', 'R_idw', 'R_fk', 'R_re_idw', 'R_re_fk']
    gcnr = np.zeros((len(x_val), len(dtypes)))
    gcnr_true = np.load('npy_data/simulations/evaluation/gcnr/R_true.npy') 
    
    for idx, item in enumerate(dtypes):
        curr_gcnr = np.load('npy_data/simulations/evaluation/gcnr/{}.npy'.format(item))
        gcnr[:, idx] = curr_gcnr/gcnr_true
    
    # Save
    # With initial sampling positions
    pgf1d.generate_coordinates_for_addplot(x_val, '{}/gcnr_smp.tex'.format(path_tex), 
                                           False, colors[:3], linestyle[:3],
                                           gcnr[:, 0], gcnr[:, 1], gcnr[:, 2])
    # With resampling positions
    pgf1d.generate_coordinates_for_addplot(x_val, '{}/gcnr_resmp.tex'.format(path_tex), 
                                           False, colors[3:], linestyle[3:],
                                           gcnr[:, 3],gcnr[:, 4])
                                           


    

def save_scan_positions_scatter_plots(coverage, setNo):
    """
    Parameters
    ----------
        coverage: int in [%]
    """
    # Setup to load the data
    path_npy = 'npy_data/simulations/210625_{}%/setNo_{}'.format(coverage, setNo)
    
    # Load
    p_smp = np.load('{}/p_smp.npy'.format(path_npy)) # shape = Ns x 2
    p_resmp_krig = np.load('{}/p_resmp_krig.npy'.format(path_npy)) # shape = Ns x 2
    p_resmp_rnd = np.load('{}/p_resmp_rnd.npy'.format(path_npy)) # shape = Ns x 2  
    rmse = np.load('{}/rmse_fk.npy'.format(path_npy)) # shape = Nx x Ny
    
    # Find the high variance points = prioritized scanning accroding to the feedback
    N_high_vari = len(np.flatnonzero(rmse >= 2* np.nanmean(rmse)))
    
    
    # Tex setup: 
    path_tex = '/Users/sayakokodera/Uni/Master/MA/tex/Defense/figures/coords_1D/simulations/{}'.format(num2string(coverage))
    # colors = ['tui_blue', 'tui_orange', 'fri_green'] 
    # styles = ['mark = *, mark size = 2pt, only marks', 'mark = square* , mark size = 2pt, only marks',
    #           'mark = square* , mark size = 2pt, only marks']
    
    # Save
    # p_smp
    pgf1d.generate_coordinates_for_addplot(p_smp[:, 0], '{}/p_smp.tex'.format(path_tex), 
                                           False, ['tui_blue'], 
                                           ['mark = *, mark size = 1pt, only marks'],
                                           p_smp[:, 1])
    # p_resmp_krig
    pgf1d.generate_coordinates_for_addplot(p_resmp_krig[:, 0], '{}/p_resmp_krig.tex'.format(path_tex), 
                                            False, ['tui_orange, opacity = 0.8'], 
                                            ['mark = asterisk, mark size = 2pt, only marks'],
                                            p_resmp_krig[:, 1])
    # p_resmp_rnd
    pgf1d.generate_coordinates_for_addplot(p_resmp_rnd[:, 0], '{}/p_resmp_rnd.tex'.format(path_tex), 
                                            False, ['fri_green, opacity = 0.8'], 
                                            ['mark = asterisk, mark size = 2pt, only marks, very thick'],
                                            p_resmp_rnd[:, 1])
    # p_high_vari
    pgf1d.generate_coordinates_for_addplot(p_resmp_krig[:N_high_vari, 0], '{}/p_high_vari.tex'.format(path_tex), 
                                            False, ['tui_red, opacity = 0.8'], 
                                            ['mark = asterisk, mark size = 2pt, only marks'],
                                            p_resmp_krig[:N_high_vari, 1])
    



    
    
    
    
    
    
    
    
    
    
    
    
    
        
    
    
    