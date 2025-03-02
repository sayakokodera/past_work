################## ESE : pos_scan tex 2D visualization #########################
import tools.tex_2D_visualization as pgf2d
import numpy as np

### labels and co ###
x_label = 'x / $\dx$'
y_label = 'y / $\dy$'
label_size = 'LARGE'
tick_size = 'LARGE'
# WGOR = white green orange red
colors = ['TUI_white', 'FRI_green_light', 'TUI_orange_light', 'TUI_red_dark']
#boundaries = np.array([0.00, 0.45, 0.55, 1.00])
# BOR = blue orange red
#colors = ['TUI_blue_dark', 'TUI_orange_light', 'TUI_red_light', 'TUI_red_dark']
boundaries = np.array([0.00, 0.45, 0.55, 1.00])
jsonFile = 'tools/TUI_FRI_colors.json'

fname_sigma = ['sigma_0_00', 'sigma_025_00', 'sigma_050_00', 'sigma_100_00', 'sigma_200_00', 'sigma_400_00']
var_sub_list = ['np_2', 'np_3', 'np_4', 'np_5']

#### reference ###
ref_cimg_max = np.load('npy_data/ESE/grid/cimg_max_05_oa_0.npy')
# get the maximal values
ref_max_cimg_max = ref_cimg_max.max()

#################################################################################################### pos_scan : w ######
# =============================================================================
#     
# for curr_var in fname_var:
# 
#     curr_npy_sum = 'npy_data/ESE/pos_scan/dim_4_posdef_0/w/np_5/cimg_sum_{}_np_3.npy'.format(curr_var)
#     curr_npy_max = 'npy_data/ESE/pos_scan/dim_4_posdef_0/w/np_5/cimg_max_{}_np_3.npy'.format(curr_var)
#     # for png files
#     curr_png_max = 'tex_figures/pos_scan/w_np_5_cimg_max_{}.png'.format(curr_var)
#     # for tex files
#     curr_tex_max = 'tex_file/tikz_posscan_w_np_5_cimg_reco_max_{}.tex'.format(curr_var)
#     
#     cimg_max = np.load(curr_npy_max)/ref_max_cimg_max
#     
#     pgf2d.create_pgf_2D(cimg_max, curr_png_max, curr_tex_max, x_label, y_label, label_size, tick_size, 
#                         custom_color = True, colors = colors, boundaries = boundaries, jsonFile = jsonFile,
#                         input_vmin_vmax = True, vmin_input = 0, vmax_input = 1)
# =============================================================================
    
################################################################################################### pos_scan : dr ######
#fpath = 'H:/2018_Sayako_Kodera_BA_Daten/npy_data/ESE/pos_scan/dim_4_posdef_0/dr'
fpath = 'npy_data/ESE/pos_scan/dim_4_posdef_0/dr'


for curr_np in var_sub_list:
    for curr_sigma in fname_sigma:
        # file names
        # for npy files
        #curr_npy_sum = '{}/{}/20181011/cimg_sum_{}.npy'.format(fpath, curr_np, curr_var)
        curr_npy_max = '{}/{}/cimg_max_{}.npy'.format(fpath, curr_np, curr_sigma)
        # for png files
        #curr_png_sum = 'tex_figures/pos_scan/dr_{}_cimg_sum_{}.png'.format(curr_np, curr_var)
        curr_png_max = 'pytikz/2D/texpngs/posscan/dr_{}_cimg_max_{}.png'.format(curr_np, curr_sigma)
        # for tex files
        #curr_tex_sum = 'tex_file/tikz_posscan_dr_{}_cimg_reco_sum_{}.tex'.format(curr_np, curr_var)
        curr_tex_max = 'pytikz/2D/posscan_dr_{}_cimg_reco_max_{}.tex'.format(curr_np, curr_sigma)
        
        #cimg_sum = np.load(curr_npy_sum)/ref_max_cimg_sum
        cimg_max = np.load(curr_npy_max)/ref_max_cimg_max
        

        pgf2d.create_pgf_2D(cimg_max, curr_png_max, curr_tex_max, x_label, y_label, label_size, tick_size, 
                            custom_color = True, colors = colors, boundaries = boundaries, jsonFile = jsonFile,
                            input_vmin_vmax = True, vmin_input = 0, vmax_input = 1)
    
