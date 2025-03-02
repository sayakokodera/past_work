# -*- coding: utf-8 -*-

import numpy as np

"""
Set of functions to modify data

"""

def remove_zero_vectors(arr, axis):
    """ Function to trim zero vectors from a 2D array
    Parameters
    ----------
        arr : np.array()
            2D array (with/without zero-vectors) 
        axis : int
            Axis to indicate whether all-zero row or column vectors are to be trimmed
            
    Returns
    -------
        Y : np.ndarray()
            2D array without zero-vectors
    """
    # X: copy of the input array and transposed if necessary s.t. its column vectors are to be trimmed
    # if column vectors are to be checked
    if axis == 0: 
        X = np.copy(arr)
    # if row vectors are to be checked -> transpose
    else:
        X = np.copy(arr).T
        
    M = X.shape[0] # Length of each vector
    N = X.shape[1] # Number of vectors to be checked
    col_toremove = []
    
    # Find zero-vectors
    for colNo in range(N):
        vec = X[:, colNo]
        if np.array_equal(vec, np.zeros(M)):
            col_toremove.append(colNo)
    
    # Remove the columns of zero-vectors
    Y = np.delete(X, col_toremove, axis = 1)
    
    if axis == 0:
        return Y
    else: 
        return Y.T 
            
    
    

