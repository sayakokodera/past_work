##### ESE Analysis for image quality ########

import numpy as np
from scipy.stats import norm
import scipy.signal as sig

        
########################################################################################################## MSE #########        
class ImageQualityAnalyzerSE():
    # constructor
    def __init__(self, reference):
        """ Constructor        
        Parameters
        ----------
            reference : 2D array (np.array(Nx, Ny)), required for ImageQualityAnlyzer
                the reference data to be compared with

        """
        super().__init__() 
        self.reference = np.array(reference)
        self.Nx = self.reference.shape[0]
        self.Ny = self.reference.shape[1]
        self.alpha = None
        self.se = None
        self.data_err = None
        
    def _check_dimension(self):
        if self.Nx == self.data_err.shape[0] and self.Ny == self.data_err.shape[1]:
            pass
        else :
            raise AttributeError('ImageQualityAnalizer : dimension of the input data and the reference do not match')
        
        
    def _calculate_se(self):
        """ calculate the MSE with alpha for bettrer scaling s.t. 0 < MSE < 1  
        """
        ref = np.array(self.reference)
        data = np.array(self.data_err)
        # calculate the alpha
        ref_vec = ref.reshape(self.Nx* self.Ny, 1)
        data_vec = data.reshape(self.Nx* self.Ny, 1)
        self.alpha = float(np.dot(np.transpose(ref_vec), data_vec) / np.dot(np.transpose(data_vec), data_vec))
        # MSE calculation
        num = np.linalg.norm(self.alpha* data - ref)
        den = np.linalg.norm(ref)
        self.se = num / den #se 
        #self.mse = np.mean(se)
        
    
    def get_se(self, data_err):
        """ with this function the mean squared error (MSE) is calculated 
        Parameters
        ----------
            data_err : 2D array (np.array(Nx, Ny))
                the data which should be compared with the given reference data
                
        Returns
        -------
            mse : float
        """
        self.data_err = np.array(data_err)/np.max(np.abs(self.reference))  
        self.reference = self.reference / np.max(np.abs(self.reference))
        self._check_dimension()
        self._calculate_se()
        return self.se


########################################################################################################## API #########        
class ImageQualityAnalyzerAPI():
    # constructor
    def __init__(self, wavelength, threshold,  dx, dy):
        """ Constructor        
        Parameters
        ----------
            reference : 2D array (np.array(Nx, Ny)), required for ImageQualityAnlyzer
                the reference data to be compared with

        """ 
        self.wavelength = wavelength # unitless [m]
        self.threshold = threshold
        self.dx = dx # unitless [m]
        self.dy = dy # unitless [m]
        self.Npix = None
        self.area = None
        self.api = None
        
    def count_pixels(self, data_raw):
        # Calculate the envelop of the data
        data = np.abs(sig.hilbert(data_raw, axis = 0))
        # Normalize the data
        data = data / np.abs(data).max()
        # Pick the elements >= threshold
        arr_bool = (np.abs(data) >= self.threshold).astype(int)
        # Count the pixels
        self.Npix = np.count_nonzero(arr_bool)
        
    def calculate_area(self):
        self.area = self.Npix* self.dx* self.dy # unitless, [m**2]
        
    def calculate_api(self, data):
        self.count_pixels(data)
        self.calculate_area()
        self.api = self.area / (self.wavelength**2)
        
    def get_api(self, data):
        self.calculate_api(data)
        return self.api
        

######################################################################################################### GCNR #########
class ImageQualityAnalyzerGCNR():
    # constructor
    def __init__(self, N_hist = 10, target_area = None):
        """ Constructor        
        Parameters
        ----------
            target_area: binary boolean array
        """
        self.N_hist = int(N_hist)
        self.target_area = target_area
        self.data = None
        self.inside = None
        self.outside = None
        self.gcnr = None
        
    def get_envelope(self, data):
        # Calculate the envelop of the reference matrix
        data_analytic = np.abs(sig.hilbert(data, axis = 0))
        # Normalize
        data_analytic = data_analytic / data_analytic.max()
        return data_analytic
        
    def set_target_area(self, reference, threshold):
        # Get envelope
        ref = self.get_envelope(reference)
        # Pick the elements >= threshold
        self.target_area = (np.abs(ref) >= threshold).astype(int) # inside = 1, outside = 0
        self.target_area = 2* self.target_area - np.ones(self.target_area.shape) # inside = 1, outside = -1
        
        
    def divide_region(self, data_raw):
        # Get envelope
        data = self.get_envelope(data_raw)
        # Devide the region: inside/outside the target area
        arr_inside = data* self.target_area
        arr_outside = data* (np.ones(self.target_area.shape) - self.target_area)
        vec_inside = arr_inside.flatten()
        vec_outside = arr_outside.flatten()
        # Keep only the nonzero values
        self.inside = vec_inside[vec_inside >= 0]
        self.outside = vec_outside[vec_outside >= 0]
        
        
    def calculate_gcnr(self, data):
        # Pick the elements inside/outside the target area
        self.divide_region(data)
        # Setting the global range
        global_max = np.maximum(np.max(self.inside), np.max(self.outside))
        global_min = np.minimum(np.min(self.inside), np.min(self.outside))
        # Calculte the histogram
        in_hist, in_binedges = np.histogram(self.inside, bins = self.N_hist, range=(global_min, global_max), 
                                            density=True)
        out_hist, out_binedges = np.histogram(self.outside, bins = self.N_hist, range=(global_min, global_max), 
                                              density=True)
        # Calculate the overlap
        ovl_hist = np.minimum(in_hist, out_hist)
        ovl_binwidth = out_binedges[1] - in_binedges[0]
        ovl = np.sum(ovl_hist * ovl_binwidth)
        self.gcnr = 1 - ovl
              
        
    def get_gcnr(self, data):
        self.calculate_gcnr(data)
        return self.gcnr
