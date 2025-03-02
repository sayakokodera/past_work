#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 00:42:59 2021

@author: sayakokodera
"""


import numpy as np


class DefectsSynthesizer():
    def __init__(self, M, Nx, Ny, range_dict):
        """
        Parameters
        ----------
        M : int
            ROI dimension along z-axis (i.e. M = Nz - Nz_offset)
        Nx : int
            ROi dimension along x-axis
        Ny : int
            ROI dimension along y-axis
        range_dict : dict
            Dictionary containing the ranges w.r.t. p_defs & amplitude
            (e.g.)
                range_dict = {
                    'x' : np.arange(5, 20), # required
                    'y' : np.arange(0, 20), # required
                    'z' : np.arange(200, 500), #required
                    'amp_low' : 0.4, # optional
                    'amp_high' : 1.0 # oprional
                    }
        """
        # Copy the parameters
        # Required keys: spatial range to select defect positions
        self.x_range = np.copy(range_dict['x'])
        self.y_range = np.copy(range_dict['y'])
        self.z_range = np.copy(range_dict['z'])
        # Optional keys: reflectivity range
        if 'amp_low' in range_dict:
            self.low = range_dict['amp_low']
        else:
            self.low = 0.0
        if 'amp_high' in range_dict:
            self.high = range_dict['amp_high']
        else:
            self.high = 1.0
        
        # Other global parameters
        self.Ndefect = None
        self.p_defs = None # defect positions, shape = (Ndefect, 3)
        self.amps = None # reflectivty (i.e. amplitude) of each defects, shape = Ndefect
        self.defmap_3d = np.zeros((M, Nx, Ny)) # defect map in 3D
        self.defmap_1d = None # unfolded defect map (= vector form)
        
        
    def select_positions(self, seed = None):
        """ Randomly select defect positions within the given ranges
        
        Parameters
        ----------
        seed : int (optional, None by default)
            seed value to initiate the random process
        """
        
        # Full mesh grid within the given range
        xx, yy, zz = np.meshgrid(self.x_range, self.y_range, self.z_range)
        p_grids = np.array([zz.flatten(), xx.flatten(), yy.flatten()]).T 
        
        # Select indices -> to choose the positions from gridded points
        rng = np.random.default_rng(seed)
        indices = rng.choice(np.arange(p_grids.shape[0]), self.Ndefect, replace = False, 
                             shuffle = False) # replace == False -> no repeats!
        
        self.p_defs = p_grids[indices, :]

        
        
    def select_reflectivity(self, seed = None):
        """ Randomly select reflectivity (i.e. amplitude) for each defect within the given ranges [self.low, self.high)
        
        Parameters
        ----------
        seed : int (optional, None by default)
            seed value to initiate the random process
        """
        rng = np.random.default_rng(seed)
        self.amps = rng.uniform(self.low, self.high, self.Ndefect)
        
        
    def set_defects(self, Ndefect, seed_p = None, seed_amp = None):
        """

        Parameters
        ----------
        Ndefect : int
            Number of defects to be generated
        
        seed_p : int (optional, None by default)
            seed value for defect positions 
        seed_amp : int (optional, None by default)
            seed value for reflectivity 

        """
        # Assign global parameters
        self.Ndefect = np.copy(Ndefect)
        
        # Select defect positions
        self.select_positions(seed_p)
        
        # Select reflectivity
        self.select_reflectivity(seed_amp)
        
        # Assign the defects to the defect map (in 3D form)
        z_defs, x_defs, y_defs = self.p_defs.T
        self.defmap_3d[z_defs, x_defs, y_defs] = self.amps
        
        # Unfold into a vector
        self.defmap_1d = self.defmap_3d.flatten('F')
        
    
    def get_defect_map_3d(self):
        return self.defmap_3d
    
    def get_defect_map_1d(self):
        return self.defmap_1d
        
        
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    M = 512
    N = 10 # = Nx = Ny
    Npulse = 128
    range_dict = {
        'x': np.arange(10),
        'y': np.arange(10),
        'z': np.arange(int(0.5* Npulse), int(M - 0.5* Npulse)),
        'amp_low' : 0.8
        }
    
    ds = DefectsSynthesizer(M, N, N, range_dict)
    ds.set_defects(100, seed_p = 10, seed_amp = 5)
    dmp_3d = ds.get_defect_map_3d()
    
    plt.imshow(np.reshape(dmp_3d, (M, N**2), 'F'))
    
    
    
        
