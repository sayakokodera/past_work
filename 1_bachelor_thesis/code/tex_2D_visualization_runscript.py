###### convert 2D array into tex-data ######
import tools.tex_2D_visualization as pgf2d
import numpy as np

notes_list = ['defmap_0', 'defmap_4', 'defmap_5', 'defmap_6']
notes_abbrev = ['oa60_dm0', 'oa60_dm4', 'oa60_dm5', 'oa60_dm6']
vari_list = ['0', '02', '05', '10', '20', '50', '100', '200']


# labels and co
x_label = 'x'
y_label = 'y'
label_size = 'LARGE'
tick_size = 'LARGE'
colors = ['TUI_blue_dark', 'TUI_blue_light', 'TUI_white', 'TUI_red_light', 'TUI_red_dark']
boundaries = np.array([0.00, 0.25, 0.50, 0.75, 1.00])
jsonFile = 'TUI_FRI_colors.json'


for curr_note, curr_abbrev in zip(notes_list, notes_abbrev):
    for curr_vari in vari_list :

        # name settings
        if curr_note == 'defmap_0':
            fname_data_zscan = 'npy_data/ESE/zscan/dim_1_grid_0_oa_60/{}/cimg_vari_{}_{}.npy'.format(curr_note, curr_vari, curr_note)
        else :
            fname_data_zscan = 'npy_data/ESE/zscan/dim_1_grid_0_oa_60/{}/cimg_vari_{}.npy'.format(curr_note, curr_vari)
        #fname_data_posscan = 
        #fname_image_posscan = 'tex_figures/ps_cimg_{}_vari_{}.png'.format(curr_abbrev, curr_vari)
        #fname_tex_posscan = 'tex_file/tikz_ps_cimg_{}_vari_{}.tex'.format(curr_abbrev, curr_vari)
        fname_image_zscan = 'tex_figures/zscan_cimg_{}_vari_{}.png'.format(curr_abbrev, curr_vari)
        fname_tex_zscan = 'tex_file/tikz_zscan_cimg_{}_vari_{}.tex'.format(curr_abbrev, curr_vari)
        
        
        # load data
        curr_cimg = np.load(fname_data_zscan)
        
        #pgf2d.create_pgf_2D(curr_cimg, fname_image_posscan, fname_tex_posscan, x_label, y_label, label_size, tick_size)
        pgf2d.create_pgf_2D(curr_cimg, fname_image_zscan, fname_tex_zscan, x_label, y_label, label_size, tick_size, 
                            custom_color = True, colors = colors, boundaries = boundaries, jsonFile = jsonFile)

