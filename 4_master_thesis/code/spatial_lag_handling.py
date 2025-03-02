#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import abc

from array_access import search_vector

#%% Base class
class SpatialLagHandling(abc.ABC):
    """ 
    Two purposes of this class:
        (1) Find unique lags within the desired range
            * Scalar case -> compute from the given raw lags
            * Vector case -> compute based on the batch size
        (2) Generate the lags_dict of the given raw lags 
            -> this allows faster access to multiple data with the same lag element
            
    Example
    -------
        slh = SpatialLagHandlingScalar(h_raw)
        h = slh.get_valid_lags()
        idx = slh.get_indices(h_tofind)
        ---> h_raw[idx[0]] == h_tofind
    """
    def __init__(self):
        self.h = None # Lags to be used to generate the lags_dictionary
        # Dictionary base
        self.lag_dict = None # Dictionary to store the lag information (lag elements, corresponding indices)
        # Valid lag indices
        self.idx_valid = None
        
    
    @abc.abstractclassmethod
    def limit_range(self):
        """
        """
    
    @abc.abstractclassmethod    
    def check_element(self):
        """
        """
        
    @abc.abstractclassmethod    
    def find_unique_lags(self):
        """
        """
        
    @abc.abstractclassmethod    
    def find_indices(self):
        """
        """
        
    def generate_lag_dictionary(self):
        """ Generate the lag dictionary containing
            * unique lags (which are within the valid range)
            * indices of the raw lags corresponding to each element of the unique lags
        Parameters
        ----------
            
        """
        # Base dictionary to store the lag information 
        self.lag_dict = {
            'lags' : self.h,
            'indices' : {} # Stores the indices of self.lags_raw to the index of each lags
            }
        
        # Find corresponding indices for each lag
        for counter, curr_lag in enumerate(self.h):
            # Indices of lags_valid corresponding to the current lag
            indices = self.find_indices(curr_lag, self.h_valid)
            # Store the information
            self.lag_dict['indices'].update({
                str(counter) : indices
                })

            
    def get_indices(self, element):
        # Check the validity of the given
        self.check_element(element)
        
        # Generate the lag_dict
        if self.lag_dict is None:
            self.generate_lag_dictionary()
               
        # Find the position of the given value in the h_unique
        lagNo = self.find_indices(element, self.h)
        
        return self.lag_dict['indices'][str(int(lagNo))]
    
    
    def get_valid_lags(self):
        return self.h
    
    def get_valid_lagindices(self):
        return self.idx_valid
        
   
    
    def get_lag_dictionary(self):
        # Generate the lag_dict
        if self.lag_dict == None:
            self.generate_lag_dictionary()
        return self.lag_dict
    

#%% Scalar-valued lags
class SpatialLagHandlingScalar(SpatialLagHandling):
    """
    A class to handle spatial information, s.a. calculate the pair-wise distances (= lags) of scan positions.
    This class delas with a scalar lags, i.e. only consider the distance of two spatial positions.
    
    Notations:
        h_ : SCALAR-valued spatial lag in [m]
    """
    
    def __init__(self, h_raw, maxlag = None, precision = 10):
        """
        Parameters
        ----------
            h_raw : np.array(N_all) in [m]
                Raw spatial lags (obtained using pdist/cdist from scipy.spatial.distance)
            maxlag : float in [m] (optional, None by default)
                Maximal spatial lag to limit the range
            precision : int (optional, 10 by default)
                Maximum number of digits (i.e. actual precision up to 10**(-precision)[m]) to be considered for
                    * finding the unique lags
                    * finding the corresponding indices to a certain lag
                      (Cf. np.isclose documentation)   
        """
        # Inherit from the parent class
        super().__init__()
        # Initial setups
        self.tolerance = 10**(-precision) #[m]
        
        # Lags to be considered (which shoud be within the valid range)
        if maxlag == None: # No limit
            self.maxlag = None
            self.h_valid = self.small_fluctuation_handling(h_raw)
        else: # With limit
            self.maxlag = np.copy(maxlag) + self.tolerance # Adding fluctuation to make the system stable
            self.h_valid, self.idx_valid = self.limit_range(h_raw, maxlag)
        # (1) Find the unique lags
        self.h = self.find_unique_lags()
        
            
    def find_unique_lags(self):
        # Lags for dictionary: the unique lags (i.e. no repeats) found in the valid lags
        return np.unique(self.h_valid[~np.isnan(self.h_valid)]) # Size = N_lag << N_all 
        

    def small_fluctuation_handling(self, h_raw):
        """ Handling the very small (< 10**-15) fluctuations in the raw lags may result in counting the same lags 
        multiple times (due to the difference of 10**-19), which we want to avoid for self.h
        
        Parameters
        ----------
            tolerance : float in [m] (10** -15 by default)
                Tolerance under which the fluctuations are ignored
        """
        
        return self.tolerance* np.around(h_raw / self.tolerance)
        

    def limit_range(self, h_raw, maxlag):
        """ Returns the lags which are smaller than the given maxlag. The ones larger than maxlag is replaced by NaN.
        (s.t. the size of the lag vectors remain same)
        
        """ 
        # Remove the ambiguity caused by the very small fluctuations in raw lags
        h_valid = self.small_fluctuation_handling(h_raw)
        # Find the indices of lags_raw larger than maxlag
        idx2remove = np.where(h_valid > maxlag)[0]
        # Indices of the lags_raw within the maxlag
        idx_valid = np.delete(np.arange(len(h_valid)), idx2remove) # shape = h_raw.shape[0] - len(idx2remove)
        # Replace the large elements with NaN 
        h_valid[idx2remove] = np.nan # shape = h_raw.shape
        
        return h_valid, idx_valid
    
    
    def check_element(self, element):
        """ Check if the given element (i.e. scalar-vauled lag) is within the valid range of <= maxlag
        """
        if self.maxlag != None and element > self.maxlag:
            raise AttributeError('SpatialLagHandling: value {} > maxlag {}!'.format(element, self.maxlag))
        else:
            pass
    
            
    def find_indices(self, val, h):
        """ Identifies the indices of lags corresponding to the given value 
        """
        return np.where(np.isclose(h, val, atol = self.tolerance))[0]
    
    
    def compute_histogram(self, bins):
        """
        Compute histogram w.r.t. the given bins including BOTH edges
        
        Parameters
        ----------
            bins : np.array(Nbins) in [m]
                Bins for computing the histogram, BOTH edges are included!
        Returns
        -------
            hist : np.array(Nbins)
                Histogram of the given datasets (i.e. lags) w.r.t. the given bins
        
        """
        # Get the available lags
        lags_avail = self.get_valid_lags() 
        
        # Compute histogram
        hist = np.zeros(bins.shape)
        for idx, item in enumerate(bins):
            if item in lags_avail:
                hist[idx] = len(self.get_indices(item))
            else:
                pass
        return hist
    
        
   
#%% Vector-valued lags
class SpatialLagHandlingVector(SpatialLagHandling):
    """
    A class to handle spatial information, s.a. computing the lags b/w two positions.
    This class deals with VECTOR-valued lags. 
    
    Notations:
        h_ : VECTOR-valued spatial lag in POLAR (!!!!!) corrdinate
        h_cart: VECTOR-valued spatial lag in cartesian corrdinate
    """
    
    def __init__(self, h_raw_cart, maxlag = None, phi_range = None, precision = 15):
        """
        Parameters
        ----------
            h_raw_cart
            maxlag
            phi_range : np.ndarray(2) in [degree] (None by default)
                Range of the angle b/w x- and y-axis, given as [phi_min, phi_max]
                The range should be [-180, 180]!
        """
        # Inherit from the parent class
        super().__init__()
        # Initial setting
        if maxlag is None:
            self.maxlag = None
        else:
            self.maxlag = np.copy(maxlag) # + 10**-(precision) -> may make it more stable
        if phi_range is None:
            self.phi_min = -180 #[deg], correspond to the min value to get with np.arctan2
            self.phi_max = 180 #[deg], correspond to the max value to get with np.arctan2
        else:
            self.phi_min, self.phi_max = phi_range
            if self.phi_min >= self.phi_max:
                raise AttributeError('SpatialLagHandlingVector: phi_max should be larger than phi_min!')
            if self.phi_min < -180 or self.phi_max > 180:
                raise AttributeError('SpatialLagHandlingVector: outside of the valid range (-180 ... 180)!')
        self.precision = np.copy(precision)
        
        # Limit the range
        self.h_valid, self.h_valid_cart = self.limit_range(h_raw_cart) # in polar & cartesian corrdinate
        # Find the unique lags
        self.h, self.h_cart = self.find_unique_lags()
        
    
    def get_valid_lags_cartesian(self):
        return self.h_cart
        
        
        
    def limit_range(self, h_raw_cart):
        """
        (1) Convert the lag into the polar corrdinate
        (2) Limit the h_raw w.r.t. the radius
        (3) Limit the h_raw w.r.t. the angle
        """
        # (1) Into the POLAR corrdinate
        h_valid = np.apply_along_axis(self.cart2pol, 1, h_raw_cart)
        # (2) Limit w.r.t. the radius (replace with NaN)
        if self.maxlag is not None:
            row2remove = h_valid[:, 0] > self.maxlag
            h_valid[row2remove, :] = np.nan
        # (3) Limit w.r.t. the angle (replace with NaN)
        row2remove = h_valid[:, 1] < self.phi_min
        h_valid[row2remove, :] = np.nan
        row2remove = h_valid[:, 1] > self.phi_max
        h_valid[row2remove, :] = np.nan
        
        # Valid lags in cartesian
        h_valid_cart = np.copy(h_raw_cart) 
        h_valid_cart[np.isnan(h_valid_cart).all(1), :] = np.nan
        
        return h_valid, h_valid_cart
        
        
    def find_indices(self, element, h):
        """
        """
        return search_vector(element, h, axis = 1)
    
        
    def check_element(self, element):
        """
        """
        pass
    
    def cart2pol(self, p):
        x, y = p
        r = np.around(np.sqrt(x**2 + y**2), self.precision)
        phi = np.rad2deg(np.arctan2(y, x))
        return np.array([r, phi])

    
    def pol2cart(self, p):
        r, phi = p
        x = np.around(r* np.cos(np.deg2rad(phi)), self.precision)
        y = np.around(r* np.sin(np.deg2rad(phi)), self.precision)
        return np.array([x, y])

    
    def find_unique_lags(self):
        """
        (1) Sort the valid lags according to the POLAR coordinate
        (2) Find the unique lags in the sorted lags (in POLAR coordinate)
        """
        # Sort according to the polar corrdinate
        h_sorted = self.sort_according2polar(self.h_valid)
        # Find the unique lags (still in POLAR coordinate)
        h_unique = np.unique(h_sorted, axis = 0)
        # get lags in the cartesian
        h_unique_cart = np.apply_along_axis(self.pol2cart, 1, h_unique)
        
        return h_unique, h_unique_cart
        
    
    def sort_according2polar(self, h_pol):
        """
        (1) Sort according to the radius
        (2) Sort according to the angle (phi)
        """
        # Sort according to the radius
        h_radsorted = h_pol[np.argsort(h_pol[:, 0]), :]
        # Sort according to the angle, phi
        h_phisorted = np.zeros(h_radsorted.shape)
        for r in np.unique(h_radsorted[:, 0]): # unique -> ascending order
            # Find the rows
            rows = np.where(h_radsorted[:, 0] == r)[0]
            # Find the elements: elms[:, 0] == r
            elms = h_radsorted[rows, :]
            # Sort the valid elements 
            elms_sorted = elms[np.argsort(elms[:, 1]), :]
            h_phisorted[rows, :] =  elms_sorted
        return h_phisorted
    

    
#%%        
if __name__ == '__main__':
    import scipy.spatial.distance as scdis
    
    def get_all_grid_points(Nx, Ny):
        xx, yy = np.meshgrid(np.arange(Nx), np.arange(Ny)) 
        return np.array([xx.flatten(), yy.flatten()]).T
    
    Nx, Ny = 3, 3
    dx = np.around(0.5, 5)
    maxlag = dx* 0.5* Nx* np.sqrt(2) + 10**-5
    
    s = dx* get_all_grid_points(Nx, Ny) # All grid points
    
    ### Scalar valued lags ###
    lags_raw = np.around(scdis.pdist(s), 10)
    lag_tofind = np.linalg.norm(s[1, :] - s[5, :])
    print('lag to find = {}'.format(lag_tofind))
    print('maxlag = {}'.format(maxlag))
    # Without maxlag
    print('*** Without maxlag ***')
    slh1 = SpatialLagHandlingScalar(lags_raw)
    lags1 = slh1.get_valid_lags()
    idx1 = slh1.get_indices(lag_tofind)
    print('Unique valid lags = {}'.format(lags1))
    print('Indices we found = {}'.format(idx1))
    
    # With maxlag
    print('*** With maxlag ***')
    slh2 = SpatialLagHandlingScalar(lags_raw, maxlag = maxlag)
    lags2 = slh2.get_valid_lags()
    idx2 = slh2.get_indices(lag_tofind)
    print('Unique valid lags = {}'.format(lags2))
    print('Indices we found = {}'.format(idx2))
    
    ### Vector valued lags ###
    print('*** Vector-valued lag ***')
    N_batch = 10
    maxlag = dx* 0.5* N_batch* np.sqrt(2) + 10**-5
    s_full = dx* (get_all_grid_points(2* N_batch, 2* N_batch) - N_batch)
    h_raw_cart = np.vstack((s_full, s_full))
    h_tofind_pol = np.array([1.58114, -71.5651]) # in POLAR coordinate
    
    slhv = SpatialLagHandlingVector(h_raw_cart, maxlag = maxlag, phi_range = np.array([-90, 90]))
    h_pol = slhv.get_valid_lags()
    idx_found = slhv.get_indices(h_tofind_pol)
    
    print('h to find = {} (in polar coord)'.format(h_tofind_pol))
    print('h_raw_pol[idx_found] = {} (in polar coord)'.format(slhv.h_valid[idx_found, :]))
    
    
    
    
    
    