""" py-pgfplots : ESE 1D visualization zscan
"""
import tools.tex_1D_visualization as vis
import numpy as np

#### MSE ###
# load data
#fpath = 'H:/2018_Sayako_Kodera_BA_Daten/npy_data/ESE/pos_scan/dim_4_posdef_0/dr/analysis'
fpath = 'npy_data/ESE/zscan/dim4_dm0_grid0/analysis'
mse_pos_oa0 = np.load('{}/mse_oa_{}_total_181212.npy'.format(fpath, 0))
mse_pos_oa1 = np.load('{}/mse_oa_{}_total_181212.npy'.format(fpath, 1))
mse_pos_oa2 = np.load('{}/mse_oa_{}_total_181212.npy'.format(fpath, 2))


# TeX settings
fname_json = 'tools/TUI_FRI_colors.json'
colors = ['TUI_blue_dark', 'TUI_orange_dark', 'FRI_green']
mark = ['', '', '']
xlabel = 'Sigma[mm]'
ylabel = 'Mean Squared Error'
fname_tex = 'pytikz/1D/mse_zscan.tex'

xvalues = mse_pos_oa0[:, 0]

vis.generate_tex_file_with_1D_plot(xvalues, fname_json, colors, mark, xlabel, ylabel, 
                                   fname_tex, mse_pos_oa0[:, 1], mse_pos_oa1[:, 1], 
                                   mse_pos_oa2[:, 1])

