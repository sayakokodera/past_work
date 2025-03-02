#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spatial subsampling for MUSE data
"""

import numpy as np

def get_all_grid_points(Nx, Ny):
        xx, yy = np.meshgrid(np.arange(Nx), np.arange(Ny)) 
        return np.array([xx.flatten(), yy.flatten()]).T
    

def batch_subsampling(Nx, N_scan, Ny = None, seed = None):
    if Ny is None:
        Ny = np.copy(Nx)
    
    # All grid points within the batch
    xx, yy = np.meshgrid(np.arange(Nx), np.arange(Ny))
    xx = xx.flatten()
    yy = yy.flatten()
    
    # Select the sampling coordinates
    rng = np.random.default_rng(seed)
    scan_idx = rng.choice(np.arange(Nx* Ny), N_scan, replace = False) # replace == False -> no repeats!
    
    coords_x = xx[scan_idx]
    coords_y = yy[scan_idx]
    coords = np.array([coords_x, coords_y]).T

    return coords


def random_walk2D(N_steps, origin, dir_set = None, seed = None):
    """
    2D random walk

    Parameters
    ----------
    N_steps : int
        Number of steps to take
    origin : np.ndarray([x, y])
        Origin point to initiate the walk.
    dir_set : np.ndarray([x_dirs, y_dirs]), optional (None by default)
        Set of possible directions for each step.
        If not specified, all 8 directions are considered, i.e. [-1, 0, 1] is used for 
        both x- and y
    seed : int (None by default)
        Seed value
        
    Returns
    -------
    path : np.ndarray((N_scan, 2)), int
        Generated path

    """
    # Initial setting
    if dir_set is None:
        # Direction of each step = all 8 possible directions
        dir_set = np.array([[1, 0], [1, 1], [0, 1], [-1, 1], 
                            [-1, 0], [-1, -1], [0, -1], [1, -1]])
        
    # Intialize the random generator
    rng = np.random.default_rng(seed)
    # Select each step randomly
    steps = rng.choice(dir_set, N_steps - 1)
    # Path = cumulative sum of the origin and the path
    path = np.concatenate((origin, steps)).cumsum(0)
    
    return path



# !!!!!!!!!! To be modified !!!!!!! include Nx AND Ny, instead of N_batch
def batch_random_walk2D(N_steps, Nx, Ny, origin = None, seed = None, ret_unique = True):
    """
    Batch spatial subsampling for e.g. imitating a manumal measurement.
    
    Two things are computed additionally in order to limit the path within the bath:
        XXXXX NO!! XXXXX(1) Direction of each step is limited to [[0, 1], [1, 0], [1, 1]]
            -> Goes only "positive" direction (no backward way to avoid the repeats)
        (2) Flipping the axis, when the path hit the boundary
            -> Keep the path within the batch
            Boundaries: 
                (1) 0 <= x, y : np.abs
                (2) x, y < N_batch : reflection
                

    Parameters
    ----------
    N_steps : int
        Number of steps
    Nx : int
        Batch size along x
    Ny : int
        Batch size along y
    origin : np.ndarray((1, 2)), optional (None by default)
        Starting point of the random path. 
        If not specified, a random point from the two lines of the batch is selected
    seed : int, optional
        Seed value to initialize the random process
    ret_unique : boolean, optional (True by default)
        True, if only the unique elements are to be returned

    Returns
    -------
    path : np.ndarray((N_steps, 2))
        Selected random walk path.

    """
    # Intialize the random generator
    rng = np.random.default_rng(seed)
    
    # Choose the starting point from two corner lines of the batch
    if origin is None:
        x_org = rng.choice(np.arange(Nx))
        y_org = rng.choice(np.arange(Ny))
        origin = np.array([[x_org, y_org]])
        
    # Random walk
    walk = random_walk2D(N_steps, origin, seed = seed)
    x = walk[:, 0]
    y = walk[:, 1]
    
    def boundary_handling(val_org, N):
        val = np.copy(val_org)
        indices = np.arange(len(val))
        # Negative part
        if val.min() < 0:
            val += abs(val.min())
        # Positive part
        if val.max() >= N:
            #print('max value is too large!')
            #print('val.min = {}, val.max = {}'.format(val.min(), val.max()))
            if val.max() - val.min() < N:
                #print('Simple offset!')
                offset = val.max() - val.min()
                val -= min(val.min(), offset)
                #print('offset = {}, val.min = {}'.format(offset, val.min()))
            else:
                #print('Deleting!!')
                if val.min() > 0:
                    #print('val.min > 0')
                    val -= val.min()
                    
                offset = int((val.max() - N)/2)
                #print('offset = {}'.format(offset))
                val -= offset
                idx_valid = np.nonzero(np.logical_and(val >= 0, val < N))[0]
                indices = np.copy(idx_valid)
                #print('val[indices].max = {}'.format(val[indices].max()))
                
        # Check
        if val[indices].min() < 0 or val[indices].max() >= N:
            print('val[indices].min = {}, val[indices].max = {}'.format(val[indices].min(), val[indices].max()))
            raise ValueError('Values are too small or too large!!!')
        
        return val, indices
    
    x, idx_x = boundary_handling(x, Nx)
    y, idx_y = boundary_handling(y, Ny)
    
    # Finf the mutual indices, in case some elements are to be deleted
    if np.array_equal(idx_x, idx_y) == True:
        idx_mutual = np.copy(idx_x)
    else:
        idx_mutual = -1*np.ones(max(len(idx_x), len(idx_y))).astype(int)
        for row in range(len(idx_mutual)):
            if row < len(idx_x):
                if idx_x[row] in idx_y:
                    idx_mutual[row] = idx_x[row]
            elif row < len(idx_y):
                if idx_y[row] in idx_x:
                    idx_mutual[row] = idx_y[row]
        rows = np.nonzero(idx_mutual >= 0)[0]
        idx_mutual = idx_mutual[rows]
    
    x = x[idx_mutual]
    y = y[idx_mutual]
    if len(x) > 0:
        #print('x.min = {}, x.max = {}, len(x) = {}'.format(x.min(), x.max(), len(x)))
        #print('y.min = {}, y.max = {}, len(y) = {}'.format(y.min(), y.max(), len(y)))
    
        walk = np.array([x, y]).T
        
        if ret_unique == True:
            return np.unique(walk, axis = 0)
        else:
            return walk
        
    else:
        if seed is None:
            walk = batch_random_walk2D(N_steps, Nx, Ny, origin = origin)
        else:
            walk = batch_random_walk2D(N_steps, Nx, Ny, origin = origin, seed = seed + 51)
        return walk


        
    
    
    


#%%
if __name__ == "__main__":

    ### Parameter Setting 
    # Variables: ROI
    zmin, zmax = 1865, 2400 # = Nt_offset, Nt, 1865....1895 = only noise ...1865, 2020
    xmin, xmax = 240, 350 
    ymin, ymax = 115, 165 
    M = zmax - zmin
    Nx = xmax - xmin
    Ny = ymax - ymin
    
    # Parameters w.r.t. batch
    N_batch = 10 # Batch size (i.e. # of pixels along each axis within a single batch)
    coverage = 0.25
    N_scan = int(coverage* N_batch**2)
    
    # Starting points of each batch
    x_start, y_start = np.meshgrid(np.arange(0, Nx, N_batch), np.arange(0, Ny, N_batch))
    x_start = x_start.flatten()
    y_start = y_start.flatten()
    
    #%% Subsampling
    np.random.seed()
    seeds = np.unique(np.random.randint(10**5, size = len(x_start) + 100))
    seeds = seeds[:len(x_start)]
    
    scan_map = np.zeros((Nx, Ny))
    
    for idx in range(len(x_start)):
        # Starting point for the current batch
        x, y = x_start[idx], y_start[idx]
        print('Batch @ (x, y) = ({}, {})'.format(x, y))
        coords = batch_subsampling(N_batch, N_scan, seed = seeds[idx])
        
        # Allocate the scan positions in map
        scan_map[x + coords[:, 0], y + coords[:, 1]] = 1
    
    # # Plots
    # import matplotlib.pyplot as plt    
    # plt.figure()
    # plt.imshow(scan_map.T)
    # plt.colorbar()
    # plt.title('Spatial subsampling: selected positions')
    # plt.xlabel('x/dx')
    # plt.ylabel('y/dy')

