r"""
1D SE data tex visualization
"""
import numpy as np
import matplotlib.pyplot as plt

import tools.tex_1D_visualization as vis

# Variable
#dtype_list = ['mse_recotrack', 'mse_recoopt', 'mse_xhat']
dtype_list = ['TLS_error_762dz', 'TLS_error_1270dz']
data = {}
# example path = npy_data/200207/uniform_762dz
date = '200207'
distribution = 'uniform'
z_idx = 762
#fnpy_path = 'npy_data/{}/{}_{}dz'.format(date, distribution, z_idx)
fnpy_path = 'npy_data/{}/{}'.format(date, distribution)


# TeX setting
ftex_path = 'tex_files'
colors = ['tui_orange', 'fri_green'] # fri_green, tui_red, tui_blue, tui_orange
mark = ['']
linestyles = ['line width = 2pt']
x_y_reverse = False

for idx, dtype in enumerate(dtype_list):
    #data = np.load('{}/{}.npy'.format(fnpy_path, dtype))
    data = 10**3*np.load('{}/{}.npy'.format(fnpy_path, dtype)) # in [mm]
    ftex = '{}/{}_{}.tex'.format(ftex_path, dtype, date)
    vis.generate_coordinates_for_addplot(data[:, 0], ftex, x_y_reverse, [colors[idx]], [linestyles[0]], 
                                         data[:, 1])
    
