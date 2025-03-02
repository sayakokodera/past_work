# -*- coding: utf-8 -*-
"""
#===========================================#
    SmartInspect raw data visualization
#===========================================#

Created on Tue Jun  9 13:24:40 2020

@author: Sayako Kodera
"""
import numpy as np
import matplotlib.pyplot as plt

from ultrasonic_imaging_python.utils.file_readers import SmartInspectReader

plt.close('all')
#======================================================================================================= Functions ====#

class SmartInspectDataFormatter():
    
    def __init__(self, p_arr, xmin = None, xmax = None, ymin = None, ymax = None):
        r""" Constructor
        
         Parameters
        ----------
            p_arr : np.ndarray
                Positional info from the SmartInspect measurements
            xmin, xmax, ymin, ymax : int (None by default)
                In this class, minimal value is included, while maximal value is excluded from the range
        """
        self.p_arr = np.array(p_arr) 
        self.xmin = self.parameter_assignment('xmin', xmin)
        self.xmax = self.parameter_assignment('xmax', xmax)
        self.ymin = self.parameter_assignment('ymin', ymin)
        self.ymax = self.parameter_assignment('ymax', ymax)
        self.x_range_pixel = tuple(np.arange(self.xmin, self.xmax))
        self.y_range_pixel = tuple(np.arange(self.ymin, self.ymax))
        self.Nx_pixel = len(self.x_range_pixel)
        self.Ny_pixel = len(self.y_range_pixel)
        self.Nt = None
        self.xy_indices = None
        self.amap = None
        
        
    def parameter_assignment(self, parameter, value):
        """
        Parameters
        ----------
            parameter: str
                Indicate which parameter to be assigend
            value: either int or None 
                Values to assign to the selected parameter
                    *int = if it is already specified 
                    *None = if it is not specified
        
        Returns
        -------
            *if value is not None, return the value
            *if vlaue is None, return the extreme value according to self.p_arr
        """
        if value is not None:
            return value
        else:
            if parameter == 'xmin':
                return int(self.p_arr[:, 0].min())
            elif parameter == 'xmax':
                return int(self.p_arr[:, 0].max()) + 1
            elif parameter == 'ymin':
                return int(self.p_arr[:, 1].min())
            elif parameter == 'ymax':
                return int(self.p_arr[:, 1].max()) + 1
        

    def find_position_index(self):
        r""" To identify the indexes which satisfy the certain positional conditions, such as y = 472 and 
        850 <= x < 900. In such case, provide xmin = 850, xmax = 900, ymin = 472, ymax = 473
                
        Returns
        -------
            xy_indices : np.ndarray(int)
                Indices of the pixel positions which satisfy the given conditions
        """    
        # Narrows the y-range
        y_idx = np.argwhere(np.logical_and(self.p_arr[:, 1] >= self.ymin, self.p_arr[:, 1] < self.ymax))[:, 0]
        # Narrows the x-range within the y-range
        self.xy_indices = y_idx[np.argwhere(np.logical_and(self.p_arr[y_idx, 0] >= self.xmin, 
                                                           self.p_arr[y_idx, 0] < self.xmax))][:, 0]
    
    
    def allocation_map(self):
        r""" Returns the "map" which indicates to which pixel positions a certain set of A-Scans should be allocated.
        
        Example
        -------
            Goal: a pixelwise data matrix within 
                x = 850...900
                y = 470...475
            -> we want to identify: which A-Scans should be extracted? where do they have to be allocated?
        
        Returns
        -------
            amap : np.ndarray 
                The allocation map corresponding to the ROI
        """
        # Check, whether the given positions are all unique (i.e. no repeats)
        if len(self.p_arr) != len(np.unique(self.p_arr, axis = 0)):
            raise ValueError('SmartInspectDataFormatter: scan positions are not unique!')
             
        self.amap = np.empty((len(self.x_range_pixel), len(self.y_range_pixel)))
        self.amap[:, :] = np.nan
        #self.amap = -1.0*np.ones((len(self.x_range_pixel), len(self.y_range_pixel)))
        for p_idx, p in enumerate(self.p_arr):
            p = p.astype(int)
            if p[0] in self.x_range_pixel:
                if p[1] in self.y_range_pixel:
                    x = self.x_range_pixel.index(p[0])
                    y = self.y_range_pixel.index(p[1])
                    self.amap[x, y] = p_idx

        
    def get_data_matrix_pixelwise(self, data):
        r""" Generate the pixelwise-gridded data matrix from the raw measurement data and the positional info provided 
        by the SmartInspect.
        
        Parameters
        ----------
            data : np.ndarray 
                Raw measurement data from SmartInspect
                
        Returns
        -------
            data_matrix_pixel : np.ndarray
                Gridded data matrix according to the traccking camera pixel
        """
        # Matrix dimension
        self.Nt = data.shape[0]
        # Base of the data matrix
        ascans_pixel = np.zeros((self.Nt, self.Nx_pixel, self.Ny_pixel))
        # Generate the allocation map
        self.allocation_map()
        # Narrow down the range
        self.find_position_index()
        # Allocate the A-Scans within the x- and y-range to the data matrix
        for idx in self.xy_indices:
            x, y = np.argwhere(self.amap == idx)[0][0], np.argwhere(self.amap == idx)[0][1]
            ascans_pixel[:, x, y] = data[:, idx]
        return ascans_pixel
 
    
    
def data_smoothing(D_raw, len_x, len_y, w):
    r""" Goal of this function is to smooth the data to reduce the measurement error and noise.
    
    Example
    -------
        
            
    Parameters
    ----------
        D_raw : np.ndarray w/ the size = Nt x Nx x Ny
            Pixelwise-gridded data matrix
        len_x, len_y : int (shoud be odd number!!!)
            Number of pixels to be considered for smoothing in each direction
            (i.e. total of len_x* len_y neighboring pixels are averaged for one pixel position)
        w: np.ndarray w/ the size len_x*len_y x 1
            Weighting vector
    
    Returns
    -------
        D_aa : np.ndarray w/ the size = Nt x Nx_pixel x Ny_pixel
            Area-averaged gridded data matrix
    """
    # Check the parameter value
    if np.mod(len_x, 2) == 0 or np.mod(len_y, 2) == 0:
        raise AttributeError('VisualizeSmartInspectRawData: length of the area should be odd numbers')
        
    # Data dimension
    Nt, Nx, Ny = D_raw.shape[0], D_raw.shape[1], D_raw.shape[2]
    
    # Adding extra rows&columns to keep the size of the output(= D_aa) same as the input(= D_raw)
    sublen_x, sublen_y = int(0.5* len_x), int(0.5* len_y)
    D_padded = np.zeros((Nt, Nx + 2* sublen_x, Ny + 2* sublen_y))
    D_padded[:, sublen_x:-sublen_x, sublen_y:-sublen_y] = np.copy(D_raw)
    
    # Function for averaging data 
    def area_averaging(curr_position):
        # Bounds
        x_l = curr_position[0] - sublen_x # Lower bound
        x_u = curr_position[0] + sublen_x + 1 # Upper bound
        y_l = curr_position[1] - sublen_y # Lower bound
        y_u = curr_position[1] + sublen_y + 1# Upper bound
        # Data in the selected area (and 1-mode unfolded)
        D_area = np.reshape(D_padded[:, x_l:x_u, y_l:y_u], (Nt, len_x*len_y), 'F')
        return np.dot(D_area, w)
    
    # Listing up all positions
    pos = np.zeros((Nx*Ny, 2)).astype(int)
    pos[:, 0] = np.stack([np.arange(Nx) for _ in range(Ny)], axis = 0).flatten('C') + sublen_x # x
    pos[:, 1] = np.repeat(np.arange(Ny), Nx) + sublen_y
    # Area_averaged data
    D_aa = np.apply_along_axis(area_averaging, 1, pos).T
    D_aa = np.reshape(D_aa, (Nt, Nx, Ny), 'F')
    return D_aa
        