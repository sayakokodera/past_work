#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def search_vector(vec, arr, axis):
    """ Function to find a vector in the given array along the desired axis.
    
    Parameters
    ----------
        vec : np.ndarray(), 1D
            A vector to be found in the array
        arr : np.ndarray(), 2D
            An array to check
        axis : int
            Axis over which the given vector is to be serached
        
    Returns
    -------
        locs : np.ndarray(), 1D, int-valued
            Locations of the array along the given axis
            (e.g.) for axis = 1
                arr[locs[0], :] == vec
    
    """
    # Find locations of the vecor in the array
    locs = np.flatnonzero(np.all(np.isclose(arr, vec), axis = axis).astype(int))
    return locs
    
    
    

