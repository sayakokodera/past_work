# -*- coding: utf-8 -*-
"""
Product-sum model for spatialtemporal variogram
"""

import numpy as np
import abc
import scipy.spatial.distance as scsd 
import scipy.optimize as sco
import skgstat as skg


class Variogram(abc.ABC):
    
    def __init__(self):
        """
        

        Returns
        -------
        None.

        """
        # Distances
        self.h_x = None
        self.h_y = None
        self.h_z = None
        # Maximal distance to consider for variogram calculation
        self.maxlag_x = None 
        self.maxlag_y = None
        self.maxlag_z = None
        # Values for measurement data
        self.values = None
        # Distance bins
        self.N_bins =  None # Number of bins
        self.bin_x = None
        self.bin_y = None
        self.bin_z = None
        # Variograms
        self._v_exp = None # Experimental variogram
        self._v_the = None # Theoretical ariogram
        self.v_raw = None
    

    def euclidean_distance(self, set1, set2):
        """ Returns the distances between every pair of two given location sets.
        
        Parameters:
        -----------
            set1 : np.ndarray, size = N1 x D
                Set of locations, D = dimension
            set2 : np.ndarray, size = N2 x D
                Set of locations, D (= dimension) should be the same as that of set1!!!!
        
        Return:
        -------
            distance matrix : np.ndarray, size = N1 x N2
                Distances as a matrix
        
        """
        D1 = set1.ndim
        D2 = set2.ndim
        
        # Check if the dimension is matched
        if D1 != D2:
            raise ValueError('Variogram euclidean distance: the dimension of the given locations should match!')
            
        # Dimension should be >= 2 for cdist 
        elif D1 == 1: 
            p1 = np.array([set1]).T 
            p2 = np.array([set2]).T 
            
        else:
            p1 = np.copy(set1)
            p2 = np.copy(set2)
        
        # Distance calculation
        return scsd.cdist(p1, p2)
 
    
    def calculate_bin(self, N_bins, maxlag):
        """ Calculate the bins for the given maximal lag with the given number of bins.
        
        Parameters
        ----------
            N_bins : int
                Number of bins
            maxlag : float 
                Maximal distance to consider for the variogram calculation
                (e.g.) max(distance)
        """
        dh = maxlag / N_bins
        return np.arange(N_bins + 1)* dh
        

    def calculate_raw_values(self):
        self.h_x = self.euclidean_distance(self._x, self._x) # Distance lags
        self.h_y = self.euclidean_distance(self._y, self._y)
        self.h_z = self.euclidean_distance(self._z, self._z)
        self._v_raw = 0.5* (self.euclidean_distance(self.values, self.values)**2)

    
    def compute_raw_variogram(self):
        """ Experimental variogram computed for "all-at-once" variogram modeling introduced by Tadic
        
        Ref
        ---
            [1] Towards Hyper-Dimensional Variography Using the Product-Sum Covariance Model 
                J. M. Tadic et al, 2019
                https://www.mdpi.com/2073-4433/10/3/148
        """
        # Raw values: distances (in x, y, z) & semi-variances 
        h_x_raw = self.euclidean_distance(self._x, self._x) # Distance lags
        h_y_raw = self.euclidean_distance(self._y, self._y)
        h_z_raw = self.euclidean_distance(self._z, self._z)
        
        # Find the positions where all maxlags are satisfied (use the same notation as matlab codes from Tadic)
        self.gd_loc = np.argwhere(np.logical_and(h_x_raw <= self.maxlag_x, 
                                                h_y_raw <= self.maxlag_y, 
                                                h_z_raw <= self.maxlag_z))
        # Distances & semi-variances corresponding to gd_loc
        self.h_x = h_x_raw[self.gd_loc]
        self.h_y = h_y_raw[self.gd_loc]
        self.h_z = h_z_raw[self.gd_loc]
        # Raw vareiogram
        self.v_raw = 0.5* (self.euclidean_distance(self.values[self.gd_loc], self.values[self.gd_loc])**2)
        
        # Remove NaN -> why???????
        
        

    def v_1D_gaussian(self, h, sill, h_inf):
        """ Gaussian model for single-axis (1D) variogram.

        Parameters
        ----------
            h : np.ndarray, ndim = 1
                Sequence of distances to model the variogram
            sill : float
                Value to be converged for h -> inf
            h_inf : float
                Distance (i.e. range) up to saturation/convergence

        Return
        ------
            Guassuain model : np.array, ndim = 1
        """
        return sill* (1 - np.exp(- h**2 / h_inf**2))
        
    
    
    @property
    def v_exp(self):
        """
        Exeperimental variogram
        """
        return self._v_exp
    
    @abc.abstractmethod
    def fit(self):
        """

        Returns
        -------
        None.

        """

    @abc.abstractmethod
    def model(self):
        """
        """
        
    
    @property
    def v_the(self):
        """
        Theoretical variogram
        """
        return self._v_the


class VariogramProductSum(Variogram):
    
    def __init__(self,
                 coordinates,
                 values,
                 N_bins,
                 maxlag = None,
                 fit_method = 'LS'
                 ):
        """
        Parameters
        ----------
            coordinates : np.ndrray, size = L x 3 (w/ L = number of all measurement values)
                Coordinates for all measurement values in x-, y- and z-axis.
                These coordinates are WITHOUT unit, i.e. pixels in all diections are treated equally.
                (e.g.)
                    coordinates = [
                        [z_{0}, x_{0}, y_{0}],
                        [z_{1}, x_{0}, y_{0}],
                            ....
                        [z_{N_voxel}, x_{0}, y_{0}],
                        [z_{0}, x_{1}, y_{0}],
                            ......
                    ]
            values : np.ndarray, size = L x 1
                All measurement values corresponding the given coordinates
            N_bins : int
                Number of distance bins for variogram 
            maxlag : np.ndarray, size = 3 x 1 (None by default)
                Maxmal distance/lag to consider for the variogram calculation
                (e.g.) 
                    maxlag = [maxlag_z, maxlag_x, maxlag_y] = [0.5* N_voxel, N_voxel, N_voxel] 
            fit_method : str ('LS' by default)
                Method name for model fitting
                'LS' = least squares
        """
        self._z, self._x, self._y = coordinates
        self.coordinates = np.copy(coordinates)
        self.values = np.copy(values)
        self.N_bins = np.copy(N_bins)
        if maxlag is None:
            self.maxlag_z = max(self._z) - min(self._z)
            self.maxlag_x = max(self._x) - min(self._x)
            self.maxlag_y = max(self._y) - min(self._y)   
        else:
            self.maxlag_z, self.maxlag_x, self.maxlag_y = maxlag
        self.fit_method = str(fit_method)
            


    def model(self, coeff):
        """
        Product-sum model for spatiltemporal (3D) variogram. Here the single-axis (1D) variogram is modeled with the Gaussian model.

        Parameers
        ---------
            coeff : np.arange, size = 9 x 1
                Coefficients 'i.e. model parameters) for product-sum model.
                These coefficients should be determined from the measurement data (by fitting of using network).
                
                coeff = [c0, c1, c2, c3, c4, c5, c6, c6, c7]
                    c0 = sill for x-axis variogram
                    c1 = range for x-axis variogram (range = distance up to saturation, denoted as h_inf)
                    c2 = sill for y-axis variogram
                    c3 = range for y-axis variogram
                    c4 = sill for z(t)-axis variogram
                    c5 = range for z(t)-axis variogram
                    c6 = sill for xy-variogram
                    c7 = sill for xyz-variogram
                    c8 = global nugget
        """
        v1_x = self.v_1D_gaussian(self.h_x, coeff[0], coeff[1])
        v1_y = self.v_1D_gaussian(self.h_y, coeff[2], coeff[3])
        v1_z = self.v_1D_gaussian(self.h_z, coeff[4], coeff[5])
        k1 = (coeff[0] + coeff[2] - coeff[6]) / (coeff[0]* coeff[2]) # weight for xy-variogram
        k2 = (coeff[6] + coeff[4] - coeff[7]) / (coeff[6]* coeff[4]) # weight for xyz-variogram
        v3 = coeff[8] + v1_x + v1_y - k1* v1_x* v1_y - k1* k2* v1_x* v1_y* v1_z
        return v3
    
    def fit(self):
        """
        """
        # Initial guess
        
        
        
    def cost_function(self, coeff):
        self._v_the = self.model(coeff)
        return np.linalg.norm((self._v_exp - self._v_the))**2
        
    

