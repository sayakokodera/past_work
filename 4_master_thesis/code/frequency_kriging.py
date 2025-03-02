#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy.fft as fft

from spatial_lag_handling import SpatialLagHandlingScalar
from spatial_subsampling import get_all_grid_points
from tools.array_access import search_vector
from signal_denoising import denoise_svd


class FrequencyKriging():
    """ Base class for frequency Kriging
    
    Notations
    ---------
    lower case = vector 
    upper case = 2+D
    _2d, _3d = dimension
    
    * Spatial-temporal (ST) domain
        a, A: data 
    * Spatial-freq. (SF) domain
        p, P: freq. response of the measurement data
        
    * FV tensor & matrix
        Psi : FV matrix(tensor) b/w the sampled positions, shape = Nf x Ns x Ns
        Gamma : FV matrix(tensor) b/w the prediction & the sampled positions, shape = Nf x Ns x Nc
    
    * Spatial variables
    S_ : set of positions
    s_: scan positions in [m]
    lag_: scalar-valued spatial lag b/w two positions in [m]
        i.e. lag_ij = sqrt((s_i - s_j)**2)

        
    * Else
    Ns : # of scans
    Nf : # of freq. bins (only positive ones, as out ST-data is real-valued)
    """
    def __init__(self):
        # Parameters
        self.h_full = None 
        self.maxlag = None
        self.grid_spacing = None
        
        # Set of positions
        self.S_smp = None # Sampled positions
        self.S_valid = None # Positions \in S_smp within the maxlag
        # Dimensions, shape
        self.N_batch = None
        self.Ns = None
        self.Nf = None
    
    
    def determine_lag_indices_vec(self, s):
        # Compute lags b/w S_valid and the given position s
        lags = np.linalg.norm((self.S_valid - s), axis = 1)
        lags = np.around(lags, self.precision)
        # Convert lags into the indices corresponding to the lags_full 
        # Indices for invalid lags (lags > maxlag) = -1
        lags_idx = -1* np.ones(self.Ns).astype(int)
        slh = SpatialLagHandlingScalar(lags, maxlag = self.maxlag)
        # Iterate over the lags_full
        for lagNo, h in enumerate(self.lags_full):
            if h in lags:
                indices = slh.get_indices(h)
                lags_idx[indices] = lagNo
        return lags_idx
        
    
    def assign_fv_values(self, fv, positions):
        # Dimension of the tensor
        N = positions.shape[0]
        # Base
        F_3d = np.max(fv, axis = 1)[:, np.newaxis, np.newaxis]* np.ones((self.Nf, self.Ns, N))
        #F_3d = np.infty* np.ones((self.Nf, self.Ns, N))
        # Assign FV values: iterate over each position
        for posNo, s in enumerate(positions):
            # Indices corresponding to lags_full
            lags_idx = self.determine_lag_indices_vec(s)
            # Find the row of valid lag indices (which are non-negative)
            rows = np.nonzero(lags_idx >= 0)[0]
            # Assign FV values
            F_3d[:, rows, posNo] = fv[:, lags_idx[rows]]
            
        return F_3d


class FrequencyKrigingPoint(FrequencyKriging):
    """ Point-wise frequency Kriging (compute a prediction for a single position)
    
    Example usage
    -------------
        pfk = FrequencyKrigingPoint(s, N_batch, lags_full)
        pfk.set_prediction_position(s0)
        p0_hat = pfk.predit()
        
    """
    
    def __init__(self, S_smp, N_batch, grid_spacing, lags_full, add_fluctuation = True, precision = 10, min_points = 6):
        # Unfold
        self.S_smp = np.copy(S_smp) # Scan positions in [m]
        self.N_batch = np.copy(N_batch) # Size of the block
        self.Nfull = self.N_batch**2
        self.grid_spacing = np.copy(grid_spacing)
        self.lags_full = np.copy(lags_full)
        self.precision = np.copy(precision)
        self.min_points = np.copy(min_points)
        
        # Maxlag
        self.maxlag = self.lags_full.max()
        if add_fluctuation == True:
            self.maxlag += 10.0**(- self.precision)
            
        # Base
        self.s0 = None # Prediction positions in [m]
        self.idx_Svalid = None # Indices of the valid positions in S_smp
        self.sigmasqrt = None # FK prediction error, shape = Nf
        self.laglambda = None # Lagrange multiplier for FK weights calculation
        
    
    def find_valid_positions(self):
        # Calculate the lags b/w s0 and sampling spotions
        lag0_smp = np.linalg.norm((self.S_smp - self.s0), axis = 1) # shape = S_smp.shape[0]
        # Identify the indices of the S_smp within the maxlag
        slh = SpatialLagHandlingScalar(lag0_smp, maxlag = self.maxlag)
        self.idx_Svalid = slh.get_valid_lagindices()
        self.Ns = len(self.idx_Svalid)
        # Limit the range: S_smp -> S_valid
        self.S_valid = self.S_smp[self.idx_Svalid, :] # shape = Ns x 2
        
        
    def set_prediction_position(self, s0):
        # Register the prediction position
        if s0.ndim == 1:
            self.s0 = s0[np.newaxis, :]
        else:
            self.s0 = np.copy(s0)
        # Limit the range: S_smp -> S_valid
        self.find_valid_positions()
         
    
    def compute_weights(self, Psi_3d, Gamma_3d, tik_factor, tikhonov = True):
        """
        Parameters (incl. global ones)
        ------------------------------
            Psi_3d : np.ndarray(Nf, N, N)
            Gamma_3d : np.ndarray(Nf, N, 1)
            tik_factor : float
        """
        ### Do Tikhonov regularization
        if tikhonov == True:
            # Transpose of Psi_3d
            PsiT_3d = np.transpose(Psi_3d, (0, 2, 1)) # shape = Nf x Ns x Ns
            # Pseudo-Tikhonov regularization
            R = np.linalg.inv(np.matmul(PsiT_3d, Psi_3d) + tik_factor* np.eye(self.Ns)) # shape = Nf x Ns x Ns
            R = np.matmul(R, PsiT_3d) # shape = Nf x Ns x Ns
            # Weights
            W = np.matmul(R, Gamma_3d) # shape = Nf x Ns x 1
            
#            # Including unbiasedness constraints: solve a \cdot x = b
#            # Construct a
#            a = np.ones((self.Nf, self.Ns+1, self.Ns+1))
#            a[:, :self.Ns, :self.Ns] = R
#            a[:, -1, -1] = 0.0
#            print('a[20, -1, -1] = {}'.format(a[20, -1, -1]))
#            print('a[20, -1, -2] = {}'.format(a[20, -1, -2]))
#            print('a[20, -2, -2] = {}'.format(a[20, -2, -2]))
#            
#            # Construct b
#            b = np.ones((self.Nf, self.Ns+1, 1))
#            b[:, :self.Ns, :] = np.copy(Gamma_3d)
#            b[:, -1, -1] = 0.0
#            print('b[20, -1, 0] = {}'.format(b[20, -1, 0]))
#            print('b[20, -2, 0] = {}'.format(b[20, -2, 0]))
#            
#            # Solve
#            x = np.matmul(np.linalg.pinv(a), b)
#            W = x[:, :-1]
#            self.laglambda = x[:, -1]
            
        # Solve a x = b for each freq. bin
        else:
            print('FK weights computation: direct LS solution')
            # Construct a
            a = np.ones((self.Nf, self.Ns+1, self.Ns+1))
            a[:, :self.Ns, :self.Ns] = np.copy(Psi_3d)
            a[:, -1, -1] = 0.0
            print('a[20, -1, -1] = {}'.format(a[20, -1, -1]))
            print('a[20, -1, -2] = {}'.format(a[20, -1, -2]))
            print('a[20, -2, -2] = {}'.format(a[20, -2, -2]))
            
            
            # Construct b
            b = np.ones((self.Nf, self.Ns+1, 1))
            b[:, :self.Ns, :] = np.copy(Gamma_3d)
            b[:, -1, -1] = 0.0
            print('b[20, -1, 0] = {}'.format(b[20, -1, 0]))
            print('b[20, -2, 0] = {}'.format(b[20, -2, 0]))
            
            # Solve
            x = np.matmul(np.linalg.pinv(a), b) # shape = Nf x (Nlag + 1) x (Nlag + 1)
            W = x[:, :-1]
            self.laglambda = x[:, -1]
            
        return np.transpose(W, (0, 2, 1))
    
    
    
    
    def predict(self, fv, P_smp_2d, tik_factor, Psi_3d = None):
        """
        Parameters
        ----------
            fv : np.ndarray(Nf, Nlag), real, positive
                Frequency variogram (function) for the selected batch
            P_smp_2d : np.ndarray(Nf, Ns_smp), complex
                Freq. response of the sampled data (for the freq. range of interest)
            tik_factor : float
                Regularization factor
            ret_3d : boolean (True by default)
            Psi_3d : np.ndarray(Nf, Ns_smp, Ns_smp), (None by default)
                FV tensor b/w the sampled positions, including the invalid lags for s0.
                This can be provided, if block FK is computed to save time.
                If not given, Psi_3d is computed on demand.
                
        """
        # Total number of freq. bins
        self.Nf = fv.shape[0] 
        # In case where there is no valid point
        if self.S_valid.shape[0] < self.min_points:
            #print('FK: not enough valid position found for s0 = {}'.format(self.s0))
            p0_hat = np.zeros(self.Nf, dtype = complex)
            self.sigmasqrt = -1.0* np.ones(self.Nf) # Squared root error of FK: negative = to be replaced!, shape = Nf
            self.laglambda = np.zeros(self.Nf) # Base for Lagrange multiplier
        
        else:
            ### Setup ###
            # Freq. response of the valid positions
            P_valid = P_smp_2d[:self.Nf, self.idx_Svalid][..., np.newaxis] # shape = Nf x Ns x 1
            
            ### (1) Construct FV tensors ###
            # Psi: FV tensor b/w the sampled positions
            if Psi_3d is None:
                Psi_3d = self.assign_fv_values(fv, self.S_valid) # shape = Nf x Ns x Ns
            elif self.Ns == 1:
                Psi_3d = Psi_3d[:, self.idx_Svalid, self.idx_Svalid][..., np.newaxis] # shape = Nf x 1 x 1
                
            else:
                Psi_3d = Psi_3d[:, self.idx_Svalid, :][:, :, self.idx_Svalid] # shape = Nf x Ns x Ns
                
            # Reduce the rank
            if self.Ns > 1:
                rank = len(np.unique(self.idx_Svalid))
                Psi_3d = denoise_svd(Psi_3d, rank)
                
                
            # Gamma: FV tensor b/w prediction and s0
            Gamma_3d = self.assign_fv_values(fv, self.s0) # shape = Nf x Ns x 1
            
            ### (2) Compute weights ###
            W0T = self.compute_weights(Psi_3d, Gamma_3d, tik_factor, tikhonov = True) # transposed, shape = Nf x 1 x Ns
            
            ### (3) Predict ###
            p0_hat = np.squeeze(np.matmul(W0T, P_valid)) # shape = Nf
            
            ### (4) Prediction error ###
            err_squared = np.squeeze(np.abs(np.matmul(W0T, Gamma_3d))) # shape = Nf
            self.sigmasqrt = self.sqrt_error(err_squared)
            
        return p0_hat
    
    
    def sqrt_error(self, err_squared):
        # Base 
        err = np.zeros(err_squared.shape)
        ### Convert to sqrt ###
        # Find where the variance is non-negative 
        idx_pos = np.nonzero(err_squared >= 0)[0]
        # Convert to sqrt
        err[idx_pos] = np.sqrt(err_squared[idx_pos])
        return err
    
        
    def prediction_error(self):
        return self.sigmasqrt
        


class FrequencyKrigingBlock(FrequencyKriging):
    """ Block-wise frequency Kriging (compute a prediction for all the unsampled positions in a batch)
    
    Example usage
    -------------
        compfk = FrequencyBlockKriging(s, N_batch, lags_full)
        
    """
    def __init__(self, S_smp, N_batch, grid_spacing, lags_full, add_fluctuation = True, precision = 10):
        # Unfold
        self.S_smp = np.copy(S_smp) # Scan positions in [m]
        self.N_batch = np.copy(N_batch) # Size of the block
        self.Nfull = self.N_batch**2
        self.grid_spacing = np.copy(grid_spacing)
        self.lags_full = np.copy(lags_full)
        self.precision = np.copy(precision)
        
        # Maxlag
        self.maxlag = self.lags_full.max()
        if add_fluctuation == True:
            self.maxlag += 10.0**(- self.precision)
            
        # Base
        self.Psi_smp_3d = None
        self.idx_smp = None
        self.idx_pred = None
        self.Nc = None
        self.sqe_pred = None # Squared error of the FK prediciton
            
            

    def find_prediction_positions(self):
        # Full scan positions in the batch
        S_full = self.grid_spacing* get_all_grid_points(self.N_batch, self.N_batch)
        S_full = np.around(S_full, self.precision)
        # Find the indices of the sampled positions in s_full 
        self.idx_smp = np.zeros(self.S_smp.shape[0]).astype(int)
        for row, s in enumerate(self.S_smp):
            self.idx_smp[row] = search_vector(s, S_full, axis = 1)
        # Prediction positions 
        self.idx_pred = np.delete(np.arange(self.Nfull), self.idx_smp) 
        self.S_pred = S_full[self.idx_pred, :]
        self.Nc = self.S_pred.shape[0] # number of prediction positions
        

        
    def point_fk(self, s0):
        # Initialize the PFK class
        pointfk = FrequencyKrigingPoint(self.S_smp, self.N_batch, self.grid_spacing, self.lags_full) 
        pointfk.set_prediction_position(s0)
        # Prediction
        p0_hat = pointfk.predict(self.fv, self.P_smp, self.tik_factor, Psi_3d = self.Psi_smp_3d) # shape = Nf
        # Prediction error
        err = pointfk.prediction_error()
        return p0_hat, err
        
        
    
    def predict(self, fv, A_smp_2d, tik_factor, f_range = None):
        """
        Parameters
        ----------
            fv : np.ndarray(Nf, Nlag), real, positive
                Frequency variogram (function) for the selected batch
            A_smp_2d : np.ndarray(Nf_fft, Ns_smp), complex
                The sampled data in time domain 
            tik_factor : float
                Regularization factor
            f_range : list of int (= [fmin, fmax]), None by default
                Freq. range corresponds to the given FV, provided as freq. bins (int) 
        Returns 
        -------
            A_hat_3d : np.ndarray([M, N_batch, N_batch])
        """
        ### Setup ###
        # Unfold
        self.fv = np.copy(fv) # Estimated FV function
        self.tik_factor = np.copy(tik_factor)
        
        # Total number of freq. bins
        self.Nf = fv.shape[0]
        M = A_smp_2d.shape[0] # Number of time 
        
        # Freq. response of the sampled data
        if f_range is None:
            fmin, fmax = 0, M
        else:
            fmin, fmax = f_range
        self.P_smp = np.fft.rfft(A_smp_2d, M, axis = 0)[fmin:fmax, :] # shape = Nf x Ns
         
        # Unscanned positions = prediction positions, S_pred
        self.find_prediction_positions()  
        
        ### (1) Construct the Psi_3d for the sampling positions ###
        self.Ns = self.S_smp.shape[0] # = Ns_smp
        self.S_valid = np.copy(self.S_smp)
        # FV tensor b/w the sampled positions
        self.Psi_smp_3d = self.assign_fv_values(fv, self.S_smp) # shape = Nf x Ns_smp x Ns_smp
        

        ### (2) FK: iterate over each prediction position ###
        out = np.apply_along_axis(self.point_fk, 1, self.S_pred)
        P_pred = out[:, 0, :].T # shape = Nf x Nc
        self.sqe_pred = np.real(out[:, 1, :]).T # shape = Nf x Nc
        
        
        ### (3) Back transform into time domain ###
        A_pred = np.fft.irfft(P_pred, n = M, axis = 0)
        
        ### (4) Combine with the available data ###
        A_hat_2d = np.zeros((M, self.Nfull))
        A_hat_2d[:, self.idx_smp] = np.copy(A_smp_2d)
        A_hat_2d[:, self.idx_pred] = np.copy(A_pred)
        
        ### (5) Convert to 3D ###
        A_hat_3d = self.array_formatting_3D(A_hat_2d)
        
        ### (6) Discard the elements with the high RMSE ###
        #rmse = self.get_prediction_error()
        #x_inaccu, y_inaccu = np.nonzero(rmse >= 5* np.mean(rmse))
        #P_hat_3d[:, x_inaccu, y_inaccu] = np.zeros((self.Nf, len(x_inaccu)))
        
        return A_hat_3d
        
        
    def array_formatting_3D(self, arr):
        return np.reshape(arr, (arr.shape[0], self.N_batch, self.N_batch), 'F')


    def get_prediction_error(self):
        # Replace the negative error values (= there are not enough neighbors) with NaN
        idx_neg = np.nonzero(self.sqe_pred[1, :] < 0)
        self.sqe_pred[:, idx_neg] = np.nan
        
        # Assign
        rmse_2d = np.zeros((self.Nf, self.Nfull))
        rmse_2d[:, self.idx_pred] = self.sqe_pred
        #Convert to 3D
        rmse_3d = self.array_formatting_3D(rmse_2d)
        
        # Root mean squared error: averaging over frequency components
        if np.all(np.isnan(rmse_3d)) == True:
            rmse = rmse_3d[0, :, :]
        else:
            #rmse = np.nanmean(rmse_3d, axis = 0)
            rmse = self.nan_handling_sum(rmse_3d)
        
        return rmse
    
    
    def nan_handling_sum(self, arr_3D):
        
        def nan2imag(arr_3D):
            # Base
            out = np.ones(arr_3D.shape, dtype = complex)* arr_3D
            # Positions of NaNs
            x, y = np.nonzero(np.isnan(arr_3D[0, :, :]) == True)
            # NaN -> 1j
            if len(x) > 0:
                out[:, x, y] = 0.0 + 1j
            return out
        
        # Replace NaNs with 1j
        arr_complex = nan2imag(arr_3D) # shape = arr_3D.shape
        # Calculate the sum along z-axis
        summed = np.sum(arr_complex, axis = 0) # shape = N_batch x N_batch
        # Find where only NaNs are (which is now represented as arr.real == 0 and arr.imag > 0)
        x, y = np.nonzero(np.logical_and(summed.real == 0, summed.imag > 0))
        if len(x) > 0:
            # Replace js with NaN, s.t NaNs remain as NaNs
            summed[x, y] = np.nan
        
        # output = real part of summed 
        return summed.real
        
        
        
        