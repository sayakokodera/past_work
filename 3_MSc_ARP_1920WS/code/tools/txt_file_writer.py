##### save data ######

import numpy as np

def write_txt_file(data, dimension, file_name):
    r"""
    this funtion serves to save a 3D array as a txt file.
    
    Parameters
    ----------
        data : np.ndarray[N, M, K] 3D!!!
        dimension : str,
            describing the detail about dimension to put in the .txt-file
            e.g.) 'Nz, Nx, Ny' in this case N = Nz, M = Nx, K = Ny
        file_name : string (should include path)
        
    """
    copy_data = np.array(data)
    
    if file_name == None:
            raise AttributeError('write_txt_file : File name should be set to save data.')
    else :        
        if copy_data.ndim == 2 :
            np.savetxt(file_name, copy_data)
                
        elif copy_data.ndim == 3 :
            f = open(file_name, 'w')
            f.write('# Array in shape: ({}) = {}\n'.format(dimension, copy_data.shape))
            for slice_data in copy_data:
                np.savetxt(f, slice_data)
                # add comment for better visualization
                f.write('# New slice \n')  
            f.close()
        
        else : 
            raise TypeError('write_txt_file : dimension of the data should be 2D or 3D.')


