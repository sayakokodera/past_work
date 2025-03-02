###### convert 2D array into tex-data ######
import numpy as np

import tools.tex_2D_visualization as pgf2d
import tools.array2image as arr2im

# labels and co
x_label = 'x'
y_label = 'y'
label_size = 'LARGE'
tick_size = 'LARGE'
colors = ['TUI_blue_dark', 'TUI_blue_light', 'TUI_white', 'TUI_orange_light', 'TUI_orange_dark']
boundaries = np.array([0.00, 0.43, 0.50, 0.57, 1.00])
jsonFile = 'tools/TUI_FRI_colors.json'

# file names
datatype = 'BScan_track_190815_lambda'#BScan_auto_190815
fdata = 'npy_data_storage/BScan/{}.npy'.format(datatype)
fpng = 'tex_pngs/{}.png'.format(datatype)
ftex = 'tex_pngs/{}.tex'.format(datatype)


# Select the image range of the array
zmin = 50
zmax = 140
# Load data
data = np.load(fdata)
data_max = abs(data).max()
# Normalize the data -> adjsut to the image range
data = data / data_max
data_img = data[zmin:zmax, :]


arr2im.get_image(data_img, colors, boundaries, jsonFile, fpng, input_vmin_vmax = False)


#pgf2d.create_pgf_2D(curr_cimg, fname_image_posscan, fname_tex_posscan, x_label, y_label, label_size, tick_size)
#pgf2d.create_pgf_2D(data_img, fpng, ftex, x_label, y_label, label_size, tick_size, 
#                    custom_color = True, colors = colors, boundaries = boundaries, jsonFile = jsonFile)

