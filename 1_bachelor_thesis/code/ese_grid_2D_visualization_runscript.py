###### convert 2D array into tex-data for 'grid' ######
import tools.tex_2D_visualization as pgf2d
import numpy as np

# labels and co
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


############################################################################################################ grid ######
fname_var = ['oa_0', 'oa_1', 'oa_2']
oa_list = ['20', '15', '10']

# reference
ref_reco_sum = np.load('npy_data/ESE/grid/cimg_sum_05_oa_0.npy')
ref_reco_max = np.load('npy_data/ESE/grid/cimg_max_05_oa_0.npy')
ref_data_sum = np.load('npy_data/ESE/grid/cimg_sum_05_oa_0.npy')
ref_data_max = np.load('npy_data/ESE/grid/cimg_max_05_oa_0.npy')
# get the maximal values
ref_max_reco_sum = ref_reco_sum.max()
ref_max_reco_max = ref_reco_max.max()
ref_max_data_sum = ref_data_sum.max()
ref_max_data_max = ref_data_max.max()


for curr_var in fname_var:
    fname_cimg_reco_sum = 'npy_data/ESE/grid/cimg_sum_05_{}.npy'.format(curr_var)
    fname_cimg_reco_max = 'npy_data/ESE/grid/cimg_max_05_{}.npy'.format(curr_var)
    fname_cimg_data_sum = 'npy_data/ESE/grid/data_cimg_sum_05_{}.npy'.format(curr_var)
    fname_cimg_data_max = 'npy_data/ESE/grid/data_cimg_max_05_{}.npy'.format(curr_var)
    
    # load data    
    reco_sum = np.load(fname_cimg_reco_sum)/ref_max_reco_sum
    reco_max = np.load(fname_cimg_reco_max)/ref_max_reco_max
    data_sum = np.load(fname_cimg_data_sum)/ref_max_data_sum
    data_max = np.load(fname_cimg_data_max)/ref_max_data_max
    
    # file names of png images
    fpng_reco_sum = 'tex_figures/grid/cimg_reco_sum_{}.png'.format(curr_var)
    fpng_reco_max = 'tex_figures/grid/cimg_reco_max_{}.png'.format(curr_var)
    fpng_data_sum = 'tex_figures/grid/cimg_data_sum_{}.png'.format(curr_var)
    fpng_data_max = 'tex_figures/grid/cimg_data_max_{}.png'.format(curr_var)
    
    # file names of tex files
    ftex_reco_sum = 'tex_file/tikz_grid_cimg_reco_sum_{}.tex'.format(curr_var)
    ftex_reco_max = 'tex_file/tikz_grid_cimg_reco_max_{}.tex'.format(curr_var)
    ftex_data_sum = 'tex_file/tikz_grid_cimg_data_sum_{}.tex'.format(curr_var)
    ftex_data_max = 'tex_file/tikz_grid_cimg_data_max_{}.tex'.format(curr_var)
    
    # reco                        
    pgf2d.create_pgf_2D(reco_sum, fpng_reco_sum, ftex_reco_sum, x_label, y_label, label_size, tick_size, 
                        custom_color = True, colors = colors, boundaries = boundaries, jsonFile = jsonFile,
                        input_vmin_vmax = True, vmin_input = 0, vmax_input = 1)
    pgf2d.create_pgf_2D(reco_max, fpng_reco_max, ftex_reco_max, x_label, y_label, label_size, tick_size, 
                        custom_color = True, colors = colors, boundaries = boundaries, jsonFile = jsonFile,
                        input_vmin_vmax = True, vmin_input = 0, vmax_input = 1)
    # data
    pgf2d.create_pgf_2D(data_sum, fpng_data_sum, ftex_data_sum, x_label, y_label, label_size, tick_size, 
                        custom_color = True, colors = colors, boundaries = boundaries, jsonFile = jsonFile,
                        input_vmin_vmax = True, vmin_input = 0, vmax_input = 1)
    pgf2d.create_pgf_2D(data_max, fpng_data_max, ftex_data_max, x_label, y_label, label_size, tick_size, 
                        custom_color = True, colors = colors, boundaries = boundaries, jsonFile = jsonFile,
                        input_vmin_vmax = True, vmin_input = 0, vmax_input = 1)
