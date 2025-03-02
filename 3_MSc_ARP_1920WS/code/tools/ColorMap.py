""" 
 " Create a Colormap Function "
This function provides a colormap with the desired colrs and gradation.
"""

import matplotlib.pyplot as plot
from .json2colorList import useSelectedColorsFunc
   
# generate a color map
     
" general colormap function "
#colors = list[N][3], corresopond to rgb values
#boundaries = numpy.array(N, 1) with the value in range 0...1, shows where color change occurs
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

    plot.register_cmap(name = '', data = cdict)
    cmap = plot.get_cmap('')
    return cmap    
    

