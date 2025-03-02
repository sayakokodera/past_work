####### load txt data #########
import numpy as np
import re

def load_txt_data(file_name):
    r"""
    with this function, saved txt data (3D) can be read and loaded as an array
    
    Parameters
    ----------
        file_name : string (should include path)
        
    Returns
    -------
        data : np.array[Nzreco, Nxreco, Nyreco] or 2D np.array
    """
    f = open(file_name, 'r')
    first_line = f.readline()
    arr_shape_list = re.findall('\d+', first_line)
    
    # case 3D 
    if len(arr_shape_list) == 3:
        Nz = int(arr_shape_list[0])
        Nx = int(arr_shape_list[1])
        Ny = int(arr_shape_list[2])        
        data = np.loadtxt(file_name).reshape((Nz, Nx, Ny))
        
    # case 2D
    else :
        data = np.loadtxt(file_name)
        
    
    return data