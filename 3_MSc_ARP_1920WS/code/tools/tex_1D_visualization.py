""" Function for converting scatter plot into pgfplot """
"""
Parameters
x_values :
    np.array((M, 1))
colors : as list
    colors selected from json file and converted to a list with json2colorList,
    with the size of (N, 1)
mark :
    list with the size of (N,1), confer pgfplot manual which can be generated with the function markStyle
*args :
    corresponds to y-values(= np.array((M,1))) for plot
    number of argumens = N
    
"""
import numpy as np
import tools.json2colorList as jsoncol

def markStyle(*args):
    mark_style = []
    for arg in args:
        mark_style.append(arg)
    return mark_style

def write_file(file_name_tex, script):
    f = open(file_name_tex, 'w')
    f.write(script)
    f.close() 

def generate_tex_file_with_1D_plot(x_values, json_file, colors, mark, xlabel, ylabel, file_name, *args):
    
    ### color setting ###
    # color list
    color_list = jsoncol.useSelectedColorsFunc(json_file, colors)
    # set script for defining color in latex
    rgb = np.zeros((len(color_list), 3))
    define_colors = ''
    for j in range(len(color_list)):
        rgb[j, :] = [round(255*i, 0) for i in color_list[j]]
        RGB = str(rgb[j, 0]) + ', ' + str(rgb[j, 1]) + ', ' + str(rgb[j, 2])
        clr_name = 'clr' + str(j+1)
        define_colors += r"""\definecolor{{{0}}}{{RGB}}{{{1}}}
                        """.format(clr_name, RGB)
    
    ### input value setting ###
    y_values = []
    for arg in args:
        y_values.append(arg)
    
    
    addplot = ''
    
    for idx_plot in range(len(y_values)):
        coord = ''
        for idx_x in range(len(x_values)):
            str_x = str(round(x_values[idx_x], 3))
            str_y = str(y_values[idx_plot][idx_x])#str(round(y_values[idx_plot][idx_x], 2))            
            # str_x, str_y ----> str_coord
            coord += r""" ({0}, {1})
                     """.format(str_x, str_y)
            #mark can be list and used different style for each plot
        
        text = r"""\addplot[clr{0}, mark = {1}] coordinates{{
                     {2}
              }}; 
              """.format(str(int(idx_plot+1)), mark[idx_plot], coord)
        addplot += text
        
    
    script = r"""\begin{{tikzpicture}}
            {0}
            \begin{{axis}}[
                xlabel = {{{1}}},
                ylabel = {{{2}}}
            ]
            {3}         
            \end{{axis}}
            \end{{tikzpicture}}
            """.format(define_colors, xlabel, ylabel, addplot)
             
    f = write_file(file_name, script)
       

def generate_tex_file_with_scatter_plot(x_values, json_file, colors, mark, xlabel, ylabel, file_name, *args):
    
    ### color setting ###
    # color list
    color_list = jsoncol.useSelectedColorsFunc(json_file, colors)
    # set script for defining color in latex
    rgb = np.zeros((len(color_list), 3))
    define_colors = ''
    for j in range(len(color_list)):
        rgb[j, :] = [round(255*i, 0) for i in color_list[j]]
        RGB = str(rgb[j, 0]) + ', ' + str(rgb[j, 1]) + ', ' + str(rgb[j, 2])
        clr_name = 'clr' + str(j+1)
        define_colors += r"""\definecolor{{{0}}}{{RGB}}{{{1}}}
                        """.format(clr_name, RGB)
    
    ### input value setting ###
    y_values = []
    for arg in args:
        y_values.append(arg)
    
    
    addplot = ''
    
    for idx_plot in range(len(y_values)):
        coord = ''
        for idx_x in range(len(x_values)):
            str_x = str(round(x_values[idx_x], 3))
            str_y = str(y_values[idx_plot][idx_x])#str(round(y_values[idx_plot][idx_x], 2))            
            # str_x, str_y ----> str_coord
            coord += r""" ({0}, {1})
                     """.format(str_x, str_y)
            #mark can be list and used different style for each plot
        
        text = r"""\addplot[clr{0}, mark = {1}, mark size=4pt, only marks] coordinates{{
                     {2}
              }}; 
              """.format(str(int(idx_plot+1)), mark[idx_plot], coord)
        addplot += text
        
    
    script = r"""\begin{{tikzpicture}}
            {0}
            \begin{{axis}}[
                xlabel = {{{1}}},
                ylabel = {{{2}}}
            ]
            {3}         
            \end{{axis}}
            \end{{tikzpicture}}
            """.format(define_colors, xlabel, ylabel, addplot)
             
    f = write_file(file_name, script)



def generate_coordinates_for_addplot(x_values_input, file_name, x_y_reverse, colors, linestyle, *args):
    r""" this function generates a tex file containig information only about coordinates for addplot in tikzpicture. 
    Other part of scripts, such as plot color and mark types, should be defined in the latex editor. 
    
    parameters : 
        x_values_input 
            np.ndarray[size_of_data, 1]
        file_name 
            file name for the generated tex file (without .tex)
        x_y_reverse
            boolean, True for swapping x and y values
        colors: list
            Collection of color names, len(colors) should be same as the number of args
        linestyle: list
            Collection of the line styles (e.g. line width, dashed etc...)
            len(linestyle) should be same as the number of args
        args 
            corresponds to y-values(= np.ndarray[size_of_data, 1]) for plot
            number of args cam vary, i.e. arbitrary number of data (y data) can be ploted in an image
    """
    
    ### input value setting ###
    x_values = list(x_values_input)
    
    y_values = []
    for arg in args:
        y_values.append(arg) #becomes list with size of [number_of_args][size_of_data]
        
    script = ''
    
    for idx in range(len(y_values)):
        coord = ''
        for curr_x in x_values:
            str_x = str(curr_x)#str(round(curr_x, 2))
            str_y = str(y_values[idx][x_values.index(curr_x)])#str(round(y_values[which_y][x_values.index(curr_x)], 2))            
            # str_x, str_y ----> str_coord
            if x_y_reverse == True :
                coord += r""" ({}, {})
                         """.format(str_y, str_x)
            else:
                coord += r""" ({}, {})
                         """.format(str_x, str_y)
    
        text = r"""\addplot[{}, {}] coordinates{{
                     {}
              }}; 
              """.format(colors[idx], linestyle[idx], coord)
        script += text
        
    write_file(file_name, script)




def generate_coordinates_for_scatter_plot(x_y_coord, file_name):
    r""" this function generates a tex file containig information only about coordinates for scatter plot.
    
    parameters : 
        x_y_coord 
            np.ndarray[size_of_data, 2] containing [x, y]
        file_name 
            file name for the generated tex file (without .tex)
 
    """    
    coord = ''
    for idx in range(len(x_y_coord)):
        curr_x = str(x_y_coord[idx, 0])
        curr_y = str(x_y_coord[idx, 1])          
        # str_x, str_y ----> str_coord
        coord += r""" ({0}, {1})
                 """.format(curr_x, curr_y)
    
    text = r"""\addplot[add_color_here, mark = add_mark_here, mark size=4pt, only marks] coordinates{{
                {0}
           }}; 
           """.format(coord)
        
    write_file(file_name, text)


def generate_coordinates(x_values_input, file_name, x_y_reverse, *args): 
    r""" this function generates a tex file containig information only about coordinates for addplot in tikzpicture. 
    Other part of scripts, such as plot color and mark types, should be defined in the latex editor. 
    !!!!! 19.08.04 !!!!!!
    adding only coordinates appears to fail in latex -> use generate_coordinates_for_addplot instead!!!
    
    parameters : 
        x_values_input 
            np.ndarray[size_of_data, 1]
        file_name 
            file name for the generated tex file (without .tex)
        x_y_reverse : boolean
            True for swapping x and y values, defualt is False
        args 
            corresponds to y-values(= np.ndarray[size_of_data, 1]) for plot
            number of args cam vary, i.e. arbitrary number of data (y data) can be ploted in an image
    """
    
    ### input value setting ###
    x_values = list(x_values_input)
    
    y_values = []
    for arg in args:
        y_values.append(arg) #becomes list with size of [number_of_args][size_of_data]
        
    script = ''
    
    for which_y in range(len(y_values)):
        coord = ''
        for curr_x in x_values:
            str_x = str(curr_x)#str(round(curr_x, 2))
            str_y = str(y_values[which_y][x_values.index(curr_x)])#str(round(y_values[which_y][x_values.index(curr_x)], 2))            
            # str_x, str_y ----> str_coord
            if x_y_reverse == True :
                coord += r""" ({0}, {1})
                         """.format(str_y, str_x)
            else:
                coord += r""" ({0}, {1})
                         """.format(str_x, str_y)
    
        text = r"""
                     {0}
              """.format(coord)
        script += text
        
    write_file(file_name, script)
