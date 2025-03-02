################## Functions for generating Latex files ###############
import numpy as np
from . import json2colorList 
from . import array2image as arr2img

"""
With this program a part of the tikzpicture script will be created, which illustrating a 2D-array in LaTeX.
To be exact with this script you can convert a 2D array into a tex-file containing the following part.
    \addplot graphics [
                    xmin = { },
                    xmax = { },
                    ymin = { },
                    ymax = {}
                ]{{{fname_png}}};
This part can be inserted in the proper tikzpicture environmanet.

"""

def create_pgf_2D(input_2D_array, fname_img, fname_tex, x_label = None, y_label = None, label_size = None, 
                  tick_size = None, custom_color = False, colors = None, boundaries = None, jsonFile = None, 
                  x_y_range = None, input_vmin_vmax = False, vmin_input = None, vmax_input = None):
    r""" what this function should do :
        1. convert the input array into a png image
        2. create the tex-script
        3. write the file
        
        Parameters
        ----------
            x_y_range : dict, containing 'xmin', 'xmax', 'ymin', 'ymax'
                default = None -> in this case the dimension of the input array is used
            input_vmin_vmax : boolean (for array2image, get_image)
                True, unless vmin = -1, vmaix = 1
    """
    # (1) convert the array into a image
    if custom_color == True:
        if input_vmin_vmax == False:
            arr2img.get_image(input_2D_array, colors, boundaries, jsonFile, fname_img) 
        elif vmin_input == None or vmax_input == None:
            vmin_input = input_2D_array.min()
            vmax_input = input_2D_array.max()
            arr2img.get_image(input_2D_array, colors, boundaries, jsonFile, fname_img, input_vmin_vmax = True, 
                              vmin_input = vmin_input, vmax_input = vmax_input)
        else:
            arr2img.get_image(input_2D_array, colors, boundaries, jsonFile, fname_img, input_vmin_vmax = True, 
                              vmin_input = vmin_input, vmax_input = vmax_input)
        
    else :
        arr2img.save_arr_as_png_simple(fname_img, input_2D_array)
    
    # (2) create the tex script    
    if x_y_range == None :
        xmin = 0
        xmax = input_2D_array.shape[1] - 1
        ymin = 0
        ymax = input_2D_array.shape[0] - 1
    else :
        xmin = x_y_range['xmin']
        xmax = x_y_range['xmax']
        ymin = x_y_range['ymin']
        ymax = x_y_range['ymax']
    
    
    script = r"""\begin{{tikzpicture}}
            \begin{{axis}}[
                enlargelimits = false,
                axis on top = true,
                axis equal image,
                %unit vector ratio=12.5 1, % change aspect ratio, corresponding 12.5dx = 1dy
                point meta min = -1,   
                point meta max = 1,
                %colorbar,
                %colormap = % = TeXcmap can be obtained with TeXcmap
                %xlabel = {{{0}}},
                %ylabel = {{{1}}},
                %label style = {{font = \{2}}},
                tick label style = {{font = \{3}}},
                %y dir = reverse,
                %xtick = {{0, 20, ..., 100}} to customize the axis
                %xticklabel = {{0, 20, 40, 60, 80, 100}}
                ]
                %\addplot[tui_red, thick, mark = x] coordinates{{
                %}};
                \addplot graphics [
                    xmin = {4},
                    xmax = {5},
                    ymin = {6},
                    ymax = {7}
                ]{{{8}}};
            \end{{axis}}
            \end{{tikzpicture}}
            """.format(x_label, y_label, label_size, tick_size, xmin, xmax, ymin, ymax, fname_img)
            
    write_file(fname_tex, script)
    


def TeXcmap(colors, cmap_boundaries, jsonFile):
    r""" convert a color map into the TeX colormap
    
    Returns
    -------
        cmap : str
    """
    # create a color_list
    color_list = json2colorList.useSelectedColorsFunc(jsonFile, colors)
    
    #create script for TeXcmap
    cmap = '{mymap}{'
    for x in range(cmap_boundaries.shape[0]):
        cmap += 'rgb(' + str(cmap_boundaries[x]) + 'pt) = (' + str(round(color_list[x][0], 2))   \
                    + ', ' + str(round(color_list[x][1], 2)) + ', ' + str(round(color_list[x][2], 2))  + ') ; '   
    cmap += '}'
    return cmap    
    

    
def write_file(file_name_tex, script):
    f = open(file_name_tex, 'w')
    f.write(script)
    f.close() 
