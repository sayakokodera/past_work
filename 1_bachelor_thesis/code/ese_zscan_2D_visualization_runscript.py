################## ESE : zscan tex 2D visualization #########################
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

fname_oa = ['oa_0', 'oa_1', 'oa_2']
oa_list = ['20', '15', '10']

fname_vari = ['016']#['004', '008', '012', '020', '030']


#### reference ###
ref_cimg_max = np.load('npy_data/ESE/grid/cimg_max_05_oa_0.npy')
# get the maximal values
ref_max_cimg_max = ref_cimg_max.max()

#fpath = 'H:/2018_Sayako_Kodera_BA_Daten/npy_data/ESE/zscan/dim4_dm0_grid0'
fpath = 'npy_data/ESE/zscan/dim4_dm0_grid0'


for curr_oa in fname_oa:
    for curr_sigma in fname_vari:
        #file names
        fname_cimg_reco_max = '{}/{}/cimg_max_sigma_{}_1.npy'.format(fpath, curr_oa, curr_sigma)
        fpng_reco_max = 'pytikz/2D/texpngs/zscan/{}_cimg_reco_max_{}.png'.format(curr_oa, curr_sigma)
        ftex_reco_max = 'pytikz/2D/zscan_{}_cimg_reco_max_sigma_{}.tex'.format(curr_oa, curr_sigma)
        # load data
        reco_max = np.load(fname_cimg_reco_max)/ref_max_cimg_max
    
        pgf2d.create_pgf_2D(reco_max, fpng_reco_max, ftex_reco_max, x_label, y_label, label_size, tick_size, 
                            custom_color = True, colors = colors, boundaries = boundaries, jsonFile = jsonFile,
                            input_vmin_vmax = True, vmin_input = 0, vmax_input = 1)
