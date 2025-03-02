""" array to image """
import numpy as np
import matplotlib.pyplot as plt

# !!! Updated on 2022.07.06(Wed)
# matplotlib now has a dedicated class for colormaps
# In contrast to ListedColormap (another custom colormap variation), LinearSegmentedColormap generates 
# a smooth shifting ombré colormap by interpolating the RGB values inbetween the given boundaries 
from matplotlib.colors import LinearSegmentedColormap


from .json2colorList import useSelectedColorsFunc


### teminology ###
# image_name contains image format, eg. '.png'


# !!!! Updated on 2022.07.06 (Wed)
def generate_color_map(colors, boundaries, jsonFile):
    
    # check error
    if boundaries.max() > 1 or boundaries.min() < 0:
        raise ValueError('boundaries should be between 0 and 1')
    
    #create color_list
    color_list = useSelectedColorsFunc(jsonFile, colors)
    
    #create a color dictionary
    cdict = {} #{} = dictionary
    cdict['red'] = []
    cdict['green'] = []
    cdict['blue'] = []
    for i in range(0, boundaries.shape[0]):
        cdict['red'].append([boundaries[i], color_list[i][0], color_list[i][0]])
        cdict['green'].append([boundaries[i], color_list[i][1], color_list[i][1]])
        cdict['blue'].append([boundaries[i], color_list[i][2], color_list[i][2]])

    plt.register_cmap(name = 'custom_cmap', data = cdict)
    cmap = plt.get_cmap('custom_cmap')
    return cmap    
    
    
def get_image(input_arr, colors, boundaries, jsonFile, fname, input_vmin_vmax = False, vmin_input = None, 
              vmax_input = None, normalize = True):
    r"""
    Parameters
    ----------
        input_vmin_vmax : boolean
            True unless vmin = -1, vmax = 1
    """
    # normalize the array 
    if input_vmin_vmax == False :
        v_min = -1
        v_max = 1
    else :
        v_min = vmin_input
        v_max = vmax_input
    copy_arr = np.array(input_arr)
    # Normalize with its maxima
    if normalize == True:
        base = copy_arr/(abs(copy_arr).max()) # should be deleted afterwards?
    else:
        base = copy_arr
    norm = plt.Normalize(vmin = v_min, vmax = v_max)
    
    # create a colormap 
    cmap = generate_color_map(colors, boundaries, jsonFile)
    #cmap = plt.get_cmap('Oranges')
    cmap.set_bad('k')
    
    # convert the array into image 
    image = cmap(norm(base))
    
    # save image
    plt.imsave(fname, image, vmin = v_min, vmax = v_max)
    print('image saved!')
    

def save_arr_as_png_simple(file_name, input_arr, normalize = True):
    r""" with this function the input array will be saved as a png image. The default color map is applied to the image. 
    When the color map should be cahnged, use other functions. 
    
    Parameters
    ----------
        file_name : str, name of the file (incluing path)
        input_arr : np.array, 2D! 
        normalize : boolean, True for normalizing with the absolute max of the array (default)
    
    """
    copy_arr = np.array(input_arr)
    # normalization
    if normalize == True:
        copy_arr = copy_arr / abs(copy_arr).max()
    
    plt.imsave(file_name, copy_arr)
    
    
