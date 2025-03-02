""" py-pgfplots : ESE 1D visualization posscan
"""
import tools.tex_1D_visualization as vis
import numpy as np

#### MSE ###
# load data
#fpath = 'H:/2018_Sayako_Kodera_BA_Daten/npy_data/ESE/pos_scan/dim_4_posdef_0/dr/analysis'
fpath = 'npy_data/ESE/pos_scan/dim_4_posdef_0/dr/analysis'
mse_pos_np2 = np.load('{}/mse_np_{}.npy'.format(fpath, 2))
mse_pos_np3 = np.load('{}/mse_np_{}.npy'.format(fpath, 3))
mse_pos_np4 = np.load('{}/mse_np_{}.npy'.format(fpath, 4))
mse_pos_np5 = np.load('{}/mse_np_{}.npy'.format(fpath, 5))

# TeX settings
fname_json = 'tools/TUI_FRI_colors.json'
colors = ['TUI_blue_dark', 'TUI_orange_dark', 'FRI_green', 'TUI_red_dark']
mark = ['', '', '', '']
xlabel = 'Sigma[mm]'
ylabel = 'Mean Squared Error'
fname_tex = 'tex_file/tikz_mse_posscan.tex'

xvalues = mse_pos_np2[:, 0]

vis.generate_tex_file_with_1D_plot(xvalues, fname_json, colors, mark, xlabel, ylabel, 
                                   fname_tex, mse_pos_np2[:, 1], mse_pos_np3[:, 1], 
                                   mse_pos_np4[:, 1], mse_pos_np5[:, 1])
