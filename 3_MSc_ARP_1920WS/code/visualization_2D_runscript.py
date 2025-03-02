###### convert 2D array into tex-data ######
import numpy as np

import tools.tex_2D_visualization as pgf2d
import tools.array2image as arr2im


def savepng(data, defpos, datatype, e_max, zcenter, xcenter):
    # Image setting
    colors = ['TUI_blue_dark', 'TUI_blue_light', 'TUI_white', 'TUI_orange_light', 'TUI_orange_dark']
    boundaries = np.array([0.00, 0.43, 0.50, 0.57, 1.00])
    jsonFile = 'tools/TUI_FRI_colors.json'
    # Normalize the data
    data_max = abs(data).max()
    data = data / data_max
    # Select the image range of the array
    zmin = zcenter - 41 
    zmax = zmin + 90
    xmin = xcenter - 10
    xmax = xmin + 20
    data_img = data[zmin:zmax, xmin:xmax]
    # File setting
    e_str = err_string(e_max)
    if datatype == 'reco_true':
        fpng = 'tex_pngs/{}dz_{}.png'.format(defpos, datatype)
    else:
        fpng = 'tex_pngs/{}dz_{}_{}lambda.png'.format(defpos, datatype, e_str)
    # Convert the array into png
    arr2im.get_image(data_img, colors, boundaries, jsonFile, fpng, input_vmin_vmax = False)


def err_string(e_max):
    if e_max - int(e_max) == 0:
        e_str = str(int(e_max))
    else:
        e_str = '{}half'.format(int(e_max))
    return e_str


if __name__ == '__main__':
    # file names
    defpos = 1270
    #errmax = '2half' 
    e_max = e_norm
    zcenter = 100 # 91 for 571dz, 97 for 762dz, 100 for 1270dz
    xcenter = x_idx
    
    savepng(Reco_opt, defpos, 'reco_opt', e_max, zcenter, xcenter)
    savepng(Reco_track, defpos, 'reco_track', e_max, zcenter, xcenter)
    #savepng(Reco_true, defpos, 'reco_true', e_max, zcenter, xcenter)