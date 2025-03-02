""" Get color list from json file """
"""
This function opens a json file, converts it into a dictionary and 
creates color list with the colors form the json file.

"Description "

useAllColorsFunc :
This function uses all the colors in the selected json file with the given order.
    inputdata = json data with following information :
        'def' = rgb values (in range of 0...1)
        'No' = order of colors, 1...N (N = number of colors)
                ---> color with No.1 = color for the minimal value
                     color with No.N = color for the maximal value
    check_reverse = either 0 (reverse = False) or 1 (reverse = True)

useSelectedColorsFunc :
This function allows users to choose some colors form json file.
    inputdata = it should contain 'def' i.e. information on rgb-values
    which_colors = a list of the deseired colors from teh selected json file
                   they should be input according to the desired colormap order
                   e.g. ) ['TUI_orange_dark']

"""
import json

def useAllColorsFunc(inputdata, check_reverse):
    with open(inputdata) as data_file:
        rawDict = json.load(data_file)
    raw_list = sorted(rawDict['colors'].values(), reverse = check_reverse)        
    color_list = []
    for h in range(len(raw_list)):
        color_list.append(raw_list[h]['def'])    
        
    return color_list


def useSelectedColorsFunc(inputdata, which_colors):
    with open(inputdata) as data_file:
        rawDict = json.load(data_file)    
    color_list = []
    for h in range(len(which_colors)):
        color_list.append(rawDict['colors'][which_colors[h]]['def'])
    return color_list

