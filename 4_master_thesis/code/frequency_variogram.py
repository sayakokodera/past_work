#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy.fft as fft
import scipy.spatial.distance as scdis
from numpy.polynomial import Polynomial
import abc

from data_modifier import remove_zero_vectors
from spatial_lag_handling import SpatialLagHandlingScalar
from spatial_subsampling import get_all_grid_points


class FrequencyVariogram(abc.ABC):
    """ 
    Base class to compute Frequency Variogram (FV)
    
    Notations
    ---------
    lower case = vector 
    upper case = 2+D
    _2d, _3d = dimension
    
    * Spatial-temporal (ST) domain
        a, A: data 
    * Spatial-freq. (SF) domain
        p, P: freq. response of the measurement data
        q, Q: freq. response of the incremental process (i.e. pari-wise difference of the data)
        Qsq: Spatio-frequency covariance of the data (= |Q|**2)
    
    * Spatial variables
    s_: scan positions in [m]
    h_: vector-valued spatial lag b/w two positions in [m]
        i.e. h_ij = s_i - s_j
    lag_: scalar-valued spatial lag b/w two positions in [m]
        i.e. lag_ij = sqrt((s_i - s_j)**2) = np.linalg.norm(h_ij)

        
    * Else
    Ns : # of scans
    Nf : # of freq. bins (only positive ones, as out ST-data is real-valued)
    
        
        
    Parameters (for all subclasses)
    -------------------------------
        s : np.array(Ns, 2)
            Scan positions
        A : np.array() either in 2D or 3D, real
            ST measurement data (i.e. a set of A-Scans) taken at the scan positions s
            Each A-Scan should be stored along the axis = 0
                (i) 2D case: (Nt, Ns) -> no zero vectors!
                (ii) 3D case: (Nt, Nx, Ny) -> may contain zero vectors!
        Nf_org : int
            Number of freq. bins (including both positive & negative elements)  
        grid_spacing : float in [m] (None by default)
            Spatial spacing between two nearest grid points.
            * Why do we need this? 
                Calculating the pair-wise distnace using the sacn positions in [m] may cause small fluctuations 
                even for the same lags. 
                To avoid this, the scan positions can be recalculated using the known grid spacing, which results
                in better precision for lag calculations. 
        maxlag : float in [m] (optional, None by default)
            Maximal lag to limit the neighbor range for FV calculation
        f_range : [fmin, fmax] (None by default)
            Define the freq. range of interest. If not given, all freq. bins are considered
        N_batch : int (required for FrequencyVCariogramDNN class)
            Batch size 
        mask_value : float (used in FrequencyVCariogramDNN class) (-1.0 by default)
            Value to indicate the DNN that certain elements in the feature vectors are missing
            Any negative value can be selected, as Qsq >= 0
        add_fluctuation : boolean (True by default)
            True, if very small fluctuation (1 nanometre or so) should be added to the given maxlag to stabilize the 
            system (as there is always some rounding eror in python which results in instable results to limit the 
            lag range)
        
    
    """
    def __init__(self):
        self.s = None # Scan positions
        self.Ns = None # Number of scans
        self.P = None  # Spatio-frequency response of the data
        self.Qsq = None # Spatio-frequency covariance of the data (= |Q|**2)
        self.Nf = None # Number of freq. bins of interest (containing only POSITIVE bins)
        self.slh = None # SpatialLagHandling class
        self.fv = None # Frequency Variogram, size = Nf_positive x Nlag
        self.lags = None # Scalar-valued lags
        self.h = None # Vector-valued lags
        # Max values to reconstruct the FV
        self.fvmax = None # size = self.Nf
        self.maxlag = None
        
        
    def set_positions(self, s):
        """ Setting the scan positions into the selected class
        Parameters
        ----------
            see above
        """
        if self.grid_spacing is None:
            self.s = np.copy(s)
        else:
            self.s = self.grid_spacing* np.around(s/ self.grid_spacing) # Improved precision
        self.Ns = self.s.shape[0]
        
        # Check if the given sample positions are valid
        if self.maxlag is not None:
            if min(scdis.pdist(self.s)) > self.maxlag:
                raise ValueError('FV class: there is no valid points (< maxlag) in the sampled positions!')     
        
        
    def set_data(self, A, Nf_org, f_range = None):
        """ Setting the ST-measurement data into the selected class
        Parameters
        ----------
            see above
        """
        # Compute the freq. response of the ST measurement data
        self.P = self.compute_frequency_response(A, Nf_org, f_range)
        self.Nf = self.fmax - self.fmin
        # Compute the spectrum density of the incremental process -> save only Qsq to reduce the required storage
        Q = self.incremental_process(self.P)
        self.Qsq = (Q.conj()* Q).real
        #del P, Q    
    
        
    
    def data_formatting(self, data):
        """
        """
        # data is 2D -> no change
        if data.ndim == 2:
            data_formatted = np.copy(data) 
            
        # data is 3D -> unfold, remove the zero vectors
        else:
            data_2d = np.reshape(data, (data.shape[0], data.shape[1]* data.shape[2]), 'F')# size = Nf_positive x Nx x Ny
            # Remove zero-column-vectors to make the size = self.Nf x Ns
            data_formatted = remove_zero_vectors(data_2d, axis = 0) # size = self.Nf x Ns
        
        # Check the data sizes:
        if data_formatted.shape[1] != self.Ns:
            raise AttributeError('FrequencyVariogram: the data does not have the same size as the scan positions!')
            
        return data_formatted
        
        
    def compute_frequency_response(self, A, Nf_org, f_range):
        """ Compute the frequency response of the input ST data
        
        Parameters
        ----------
            A : np.ndarray()
                ST measurement data
            Nf_org : int
                Number of ALL frequency bins (both positive and negative ones)
        Returns
        -------
            P : np.ndarray(self.Nf, Ns), complex
                Freq. response of A
        """
        # Frequency response
        P = fft.rfft(A, n = Nf_org, axis = 0)
        # Limit the freq. range if necessary
        if f_range is None:
            self.fmin = 0
            self.fmax = P.shape[0]
        else:
            self.fmin, self.fmax = f_range
        # Formatting the data into the correct shape (= self.Nf x Ns)
        P = self.data_formatting(P[self.fmin:self.fmax, :])  
        return P
        
    
    
    def pairwise_difference(self, vals):
        """ Calculate the pari-wise distances of a set of values
        
        Parameters
        -----------
            vals: a vector (int, float, complex) with size = N
            
        Output
        ------
            pdiff: a vector with size = N* (N-1)/2 (binomial formula)
                pair-wise differences
        """
        # Initialization
        N = len(vals)
        arr1 = np.repeat(vals.reshape(N, 1), N-1, axis = 1)
        arr2 = np.repeat(vals[:-1].reshape(N-1, 1), N, axis = 1).T
        
        # Pair-wise difference = arr1 - arr2
        diff = arr1 - arr2
        # Replace the upper diagonals (i.e.repeats) with NaNs
        if np.any(np.iscomplex(vals)): # Complex-valued
            diff[np.triu_indices_from(diff)] = np.nan + 1j* np.nan 
        else: # Real-valued
            diff[np.triu_indices_from(diff)] = np.nan
        # Remove NaNs
        pdiff = diff.flatten('F')
        pdiff = pdiff[~np.isnan(pdiff)] 
        
        return pdiff    
    
    
    def incremental_process(self, data, axis = 1)  :
        """ Calculate the incremental process (i.e. pair-wise difference) of a set of vector-valued data
        
        Input
        -----
            data: np.ndarray(M, N) or (N, M), real/complex (2D!!)
                Data to be transformed into a incremental process
                With
                    N = # of vectors 
                    M = the size of each vector (i.e. invariant in this process)
            axis: int (1 by default)
                Axis over which the pair-wise difference to be calculated 
        
        Output
        ------
            data_pdiff np.ndarray(), 2D, real/complex
                Incremental process  (i.e. vector-valued pair-wise difference) of the given data
                size =  (i) M x K for axis == 0
                        (ii) K x M for axis == 1
                        with 
                            K = N* (N - 1)/2
                (e.g.) for axis = 0
                    data_pdiff[:, k] = data[:, i] - data[:, j]    
            
        """
        # Incremental process
        data_pdiff = np.apply_along_axis(self.pairwise_difference, axis, np.copy(data))
        
        return data_pdiff
    

    @abc.abstractmethod
    def compute_fv(self):
        """
        """
        
    @abc.abstractclassmethod
    def denormalize_fv(self):
        """
        """
    
    def get_frequency_response(self):
        return self.P
    
    def get_fv(self):
        return self.fv
    
    def get_fvmax(self):
        return self.fvmax
    
    def get_lags(self):
        if self.lags is None:
            self.compute_lags()
            self.lags = np.concatenate((np.zeros(1), self.lags))
        return np.around(self.lags, self.precision)
    
    
    def average_fv_singlelag(self, lag):
        """ Average semi-variances for single lag (base function to be used with apply_along_axis)
        """
        # Get indices corresponding to the given 
        # For SLH Scalar
        if len(lag.flatten()) == 1: 
            indices = self.slh.get_indices(float(lag)) # size = N_idx (varies with the lag value)
        # For SLH Vector
        else: 
            indices = self.slh.get_indices(lag) # size = N_idx (varies with the lag value)
        
        # FV values for a single lag = mean of the spatial coveriance self.Qsq
        fv_singlelag = np.mean(self.Qsq[:, indices], axis = 1) # size = self.Nf
        
        return fv_singlelag#.real
    
    
    
#%% Raw FV
   
class FrequencyVariogramRaw(FrequencyVariogram):
    """
    Class to calculate experimental (i.e. raw) Frequency Variogram (FV).
    
    By raw FV it means that the FV is calculated based solely on the given dataset, using its mean values for 
    each lag & freq. bin. This can be used, for instance, to generate the ML outputs (i.e. labels).
    
    Example
    -------
        fvr = FrequencyVariogramRaw(grid_spacing = dx, maxlag = maxlag)
        fvr.set_positions(s)
        fvr.set_data(a, M, f_range = [18, 50])
        fvr.compute_fv()
        fvr.smooth_fv(deg, ret_normalized = True)
        fv_norm = fvr.get_fv()
        fv = fvr.denormalize_fv(fv_norm)
    
    """
    def __init__(self, grid_spacing = None, maxlag = None, add_fluctuation = True, precision = 10):
        """
        Parameters
        ----------
            see above comments
        """
        # Inherit from the parent class
        super().__init__()
        # Initial setting
        self.grid_spacing = np.copy(grid_spacing)
        self.precision = np.copy(precision)
        if maxlag is None:
            self.maxlag == None
        else:
            self.maxlag = np.copy(maxlag) #[m]
            if add_fluctuation == True:
                self.maxlag = self.maxlag + 10.0**(- self.precision)
            

        
    def compute_lags(self):
        """ Compute scalar-valued lags using SpatialLagHandling
        """
        lags_raw = np.around(scdis.pdist(self.s), self.precision)
        self.slh = SpatialLagHandlingScalar(lags_raw, maxlag = self.maxlag)
        self.lags = self.slh.get_valid_lags() # !!! WITHOUT lag == 0 !!!!
        
        
    def compute_fv(self):
        """ Compute the raw FV (i.e. averaging the spatial covariance of the incremental process self.Q)
                
        """
        # Calculate the spatial lags
        self.compute_lags()
        
        # Calculate the average of spatial covariances for each lag
        #print('FV raw: lags.shape = {}'.format(self.lags.shape))
        self.fv = np.apply_along_axis(self.average_fv_singlelag, 0, np.array([self.lags])) # size = Nf_positive x Nlag
        
        # Concatenate zeros for lag == 0
        self.lags = np.concatenate((np.array([0]), self.lags))
        self.fv = np.concatenate((np.array([np.zeros(self.Nf)]).T, self.fv), axis = 1) 
        
    
    def smooth_fv(self, deg, ret_normalized = False, lags_fv = None):
        """Smooth the experimental frequency variogram via curve(polynomial) fitting. FV exhibits the different "shape"
        depending on the frequency, thus we compute a polynomial for each frequency component. 
        
        ####### NEW ########
        This is done by first normalizeing values with the maximal value of each freq. bin to improve the accuracy (?).
        Returned FV is normalized, but can be reconstructed using self.fvmax.  (i.e. row-wise normalization)
        #######
    
                
        Parameters
        ----------
            deg : int 
                Degree of the fitting polynomials 
            ret_normalized : boolean (optional, False by default)
                True, if the FV to be normalized (for ML labeling, for instance)
            lags_fv : np.array(N_lag) (optional, None by default)
                lags to be used to compute FV after the coefficients are determined
                ->  this means that the smoothed FV can be "interpolated" by providing 
                    different lags than the ones used to compute the raw FV 
                (e.g.) for sampled FV
                    there are only 20 unique lags available from the scanned positions, but in the batch
                    there should be 25 unique lags 
                        len(lags_sample) = 20
                        len(lags_full) = 25
                    -> "interpolating" 5 missing lags by providing lags_fv = lags_full       
    
        Output
        -------
        fv_smt : np.ndarray(self.Nf, N_lag), real & positive
            Smoothed frequency variogram 
            For a fixed freq. bin and a fixed lag, smoothed_fv is calculated as 
                smoothed_fv[f_bin, i] = c0 + c1* lag[i] + c2* lag[i]**2 + c3* lag[i]**3 + .....
            -> Meaning, for all spatial lags:
                smoothed_fv[f_bin, :] = np.dot(mat_lag, coeff):
                    mat_lag = np.array(N_lag, deg+1)
                            = np.array([
                                [1, lag[0], lag[0]**2, ..... lag[0]**deg],
                                [1, lag[1], lag[1]**2, ..... lag[1]**deg],
                                [1, lag[2], lag[2]**2, ..... lag[2]**deg],
                                ...
                            ])
                    coeff = np.array([c0, c1, c2, ....c_deg])
        """
        # Construct mat_lag
        if lags_fv is None: # = no "interpolation"
            lags_fv = np.copy(self.lags) 
        mat_lag = np.zeros((len(lags_fv), deg+1))
        for col in range(mat_lag.shape[1]):
            mat_lag[:, col] = lags_fv**col
        
        self.fvmax = np.zeros(self.Nf)
        
        ############################################################
        ### Base function to fit polynomials for each freq. bin ###
        def polynomial_fit_singlebin(f_bin):
            # Identify the maximal value for normalization
            vmax = self.fv[int(f_bin), :].max()
            self.fvmax[f_bin] = vmax
            # Normalize -> more accurate?
            if vmax != 0:
                fv_norm = self.fv[int(f_bin), :] / vmax # shape = N_lag
            else:
                fv_norm = np.copy(self.fv[int(f_bin), :]) # shape = N_lag
            # Fitting
            poly = Polynomial.fit(self.lags, fv_norm, deg = deg)
            coeff = poly.convert().coef
            return np.dot(mat_lag, coeff)
        ############################################################
        
        # Polynomial fit (obtained one is row-wise normalized)
        fv_smt = np.apply_along_axis(polynomial_fit_singlebin, 0, 
                                     np.array([np.arange(self.Nf)])).T # shape = Nf x N_lag
        # Denormalize
        if ret_normalized == False:
            fv_smt = self.denormalize_fv(fv_smt)
        # Copy
        self.fv = np.copy(fv_smt)

    
    def denormalize_fv(self, fv_normalized):
        """ Denormalize the given FV with the maximal value which is identified during the smoothing 
        
        Parameters
        ----------
            fv_normalized : np.array((Nf, N_lag)), complex
                Normalized FV w.r.t. the maximal value of each freq. component (i.e. row-wise normalization)
        """
        return fv_normalized* self.fvmax[:, np.newaxis]

    
#%% FV estimation via DNN
class FrequencyVariogramDNN(FrequencyVariogram):
    """ 
    Class to estimate the FV via DNN. 
    
    This subclass mainly does the feature mapping & normalizing, which is
        (1) Compute the pair-wise spatial lag VECTORS in polar corrdinate (= self.h_pol)
            (!!!)   Since we compute the FV on the gridded points in this class, the parameter
                    grid_spacing should be provided!!!!!
        (2) Compute the periodogram of the SF incremental process, Q, w.r.t. self.h
            (Periodogram here = average value of |Q|**2 for each h_pol and freq. bin)
        (3) Normalize the perodogram w.r.t. the maximal value of each freq. bin
        
    Example
    -------
        fvdnn = FrequencyVariogramDNN(grid_spacing = round(dx, 6), N_batch = 10, maxlag = maxlag)
        fvdnn.set_positions(np.around(s_smp, 6))
        fvdnn.set_data(a_smp, M, f_range = np.array([0, 50]))
        fv_feat = fvdnn.feature_mapping_vector_valued()
        or 
        fv_feat, hist = fvdnn.feature_mapping_scalar_valued()
        
    """
    def __init__(self, grid_spacing, N_batch, maxlag = None, precision = 10, mask_value = round(-1.0, 2),
                 add_fluctuation = True):
        """
        Parameters
        ----------
            see above comments
        """
        # Inherit from the parent class
        super().__init__()
        # Initial setting
        self.grid_spacing = np.copy(grid_spacing)
        self.N_batch = np.copy(N_batch)
        self.precision = np.copy(precision)
        if maxlag is None:
            self.maxlag = None
        else:
            self.maxlag = np.copy(maxlag)
            if add_fluctuation == True:
                self.maxlag = self.maxlag + 10.0**(- self.precision)
        
        # Features = full sclar lags
        self.features = None 
        # DNN input 2: histogram in terms of the full lags
        self.hist = None 
        # DNN parameter: Conv1D window length
        self.l_cnnwin = None 
        self.f_offset = None # Offset to adjust to the Conv1D window length
        
        # Initialize the Raw FV class
        self.cfvraw = FrequencyVariogramRaw(grid_spacing = self.grid_spacing, maxlag = self.maxlag)
        
        
    def compute_features(self):
        """ 
        Compute the features of our DNN model.
        Features of this DNN models are unique lag vectors of a single batch in POLAR coordinate. 
            *** Why?
                In our UT measurment setup, the incremental processes are assumed to be WSS w.r.t. spatial lag vectors.
                Meaning, Qsq (= |Q|**2) is only direction dependent (and Qsq remains the same whether the corresponding
                lag vector is \bm{h} or - \bm{h}).
        """
        ### Features = scalar-valued lags of all grid points ###
        # Find the unique scalar lags
        # Full grid points
        s_full = np.around(self.grid_spacing* get_all_grid_points(self.N_batch, self.N_batch), 10)
        # Compute the scalar lags for s_full
        lags_raw_full = np.around(scdis.pdist(s_full), self.precision)
        slh = SpatialLagHandlingScalar(lags_raw_full, self.maxlag)
        # Get the valid lags = features
        self.features = slh.get_valid_lags() # shape = Nlag -1
        # Add lag = 0
        self.features = np.concatenate((np.zeros(1), self.features)) # shape = Nlag
        # Total number of possible lags, when fully scanned -> for p.d.f. calculation
        self.N_alllags = len(lags_raw_full) + self.N_batch**2
 
    
    def get_features(self):
        return self.features
        
    
    def compute_histogram(self):
        """ Compute the histogram of the available raw scalar lags
        """
        if self.features is None:
            self.compute_features()
            
        if self.slh is None:
            lags_raw = np.around(scdis.pdist(self.s), self.precision)
            self.slh = SpatialLagHandlingScalar(lags_raw, self.maxlag)
            self.h = self.slh.get_valid_lags()
        # Get histogram from the SLH class
        hist = self.slh.compute_histogram(self.features)  # shape = Nlag
        # Assign the number of scans for lag == 0
        hist[0] = self.s.shape[0]
        # Scaling relative to fully gridded case 
        hist_scaled = hist / self.N_alllags
        
        return hist_scaled   
    

    def extend_frequency_bins(self, arr, l_cnnwin, extend_onlyneg = True):
        """ Extend freq. bins by int(l_cnnwin/2) in the negative freq. components
        (e.g.) l_cnnwin = 5 => extensions by 2 bins
        """
        ### Negative component ###
        # Freq. responses to add (in the negative freq. direction)
        extend_neg = arr[1:int(l_cnnwin/2) + 1]
        # Flip 
        extend_neg = np.flip(extend_neg, axis = 0)
        # Extend
        arr_extended = np.concatenate((extend_neg, arr), axis = 0)
        
        ### Positive component ###
        if extend_onlyneg == False:
            # Freq. responses to add (in the positive freq. direction)
            extend_pos = arr[-int(l_cnnwin/2) + 1:-2]
            # Flip 
            extend_pos = np.flip(extend_pos, axis = 0)
            # Extend
            arr_extended = np.concatenate((arr_extended, extend_pos), axis = 0)
        
        return arr_extended              
    
    
    def feature_mapping(self, data, Nf_org, l_cnnwin, deg, f_range = None):
        """
        (1) Compute the features = valid scalar lags of full grid points
        (2) Compute the lag_dict & histogram using SLH scalar
        (3) Compute the relateive histogram
        (4) Compute the smoothed raw FV using FrequencyVariogramRaw
        (5) Adjust the freq. range for CNN1D
        
        Parameter
        ---------
            l_cnnwin : int
                Window length for Conv1D net
        """
        # Unroll the parameters 
        self.l_cnnwin = l_cnnwin
        # Freq. range: adjusted to teh Conv1D window length
        self.f_offset = int(self.l_cnnwin/2)
        
            
        ### (1) Compute the features = full lags ###
        if self.features is None:
            self.compute_features()
        ### (2) Compute the lag_dict -> available valid lags ###
        if self.slh is None:
            lags_raw = np.around(scdis.pdist(self.s), self.precision)
            self.slh = SpatialLagHandlingScalar(lags_raw, self.maxlag)
            self.h = self.slh.get_valid_lags()
        
        ### (3) Compute the histogram ###
        self.hist = self.compute_histogram()       
        
        ### (4) Compute raw FV for entire ferq. range (i.e. using the original freq. range) ###
        self.cfvraw.set_positions(self.s) #Don't forget ro multiply with dx!!!!!!! 
        self.cfvraw.set_data(data, Nf_org) # WITHOUT specifying the freq rage!
        self.cfvraw.compute_fv()
        self.cfvraw.smooth_fv(deg, ret_normalized = True, lags_fv = self.features)
        fv_rawsmt = self.cfvraw.get_fv() # shape = Nf_org x 25
        fvmax_raw = np.max(self.cfvraw.denormalize_fv(fv_rawsmt), axis = 1) # shape = Nf_org
        
        ### (5) Adjust the freq. range for window length of CNN1D -> 3 cases ###
        # Case(1): entire freq. range = 0...f_Ny -> artifical extension for both negative & positive side
        if f_range is None:
            self.fmin, self.fmax = 0, Nf_org
            # Freq. extension should also be applied to the components larger than f_Ny
            self.fv_DNN = self.extend_frequency_bins(fv_rawsmt, self.l_cnnwin, extend_onlyneg = False)
            
        # Case(2): fmin == 0 -> artifical extension of negative components required for CNN1D
        elif f_range[0] == 0.0:
            self.fmin, self.fmax = f_range
            # Specify the freq. range for CNN1D
            fmin_DNN = 0
            fmax_DNN = self.fmax + int(l_cnnwin/2)
            # Extension only applied to the negative side
            self.fv_DNN = self.extend_frequency_bins(fv_rawsmt[fmin_DNN: fmax_DNN, :], self.l_cnnwin)
            
        # Case(3): fmin != 0 -> no artifical extension required
        else:
            self.fmin, self.fmax = f_range
            # Specify the freq. range for CNN1D
            fmin_DNN = self.fmin - int(l_cnnwin/2)
            fmax_DNN = self.fmax + int(l_cnnwin/2)
            self.fv_DNN = np.copy(fv_rawsmt[fmin_DNN : fmax_DNN, :])
            
        # Number of total freq. component
        self.Nf = self.fmax - self.fmin
        # fvmax
        self.fvmax = fvmax_raw[self.fmin : self.fmax]
    
        
    def get_DNNinputs(self, f_bin):
        # Select the neighboring bins to adjust the input for Conv1D net
        f_windows = np.arange(f_bin - self.f_offset, f_bin + self.f_offset + 1) + self.f_offset
        input_fv = self.fv_DNN[f_windows, :]
        # Format the histogram as Conv1D input
        input_hist = np.tile(self.hist, (self.l_cnnwin, 1)) # shape = l_cnnwin x 25
        return input_fv, input_hist
                 
            
    def compute_fv(self, model):
        
        def predict_fv_singlebin(f_bin):
            # Get the DNN inputs
            input_fv, input_hist = self.get_DNNinputs(f_bin)
            # Prediction
            y_pred = model.predict([input_fv[np.newaxis, :, :], input_hist[np.newaxis, :, :]])
            return y_pred
        
        # Normalized FV
        self.fv = np.apply_along_axis(predict_fv_singlebin, 0, np.array([np.arange(self.Nf)])).T # Nf x Nlag x 1
        self.fv = self.fv[..., 0]
        

    def denormalize_fv(self, fv_normalized):
        return fv_normalized* self.fvmax[:, np.newaxis]
        
        
        



    
        