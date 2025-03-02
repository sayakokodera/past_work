#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.interpolate import interp2d

from spatial_lag_handling import SpatialLagHandlingScalar
from spatial_subsampling import get_all_grid_points
from tools.array_access import search_vector


class BatchInterpolation3D():
    
    def __init__(self, S_smp, N_batch, grid_spacing, maxlag, add_fluctuation = True, precision = 10):
        # Unfold
        self.S_smp = np.copy(S_smp) # Scan positions in [m]
        self.N_batch = np.copy(N_batch) # Size of the block
        self.Nfull = self.N_batch**2
        self.grid_spacing = np.copy(grid_spacing)
        self.precision = np.copy(precision)
        
        # Maxlag
        self.maxlag = np.copy(maxlag)
        if add_fluctuation == True:
            self.maxlag += 10.0**(- self.precision)
            
        # Base
        self.idx_smp = None
        self.idx_pred = None
        self.S_pred = None
        self.Ns = None
        self.Nc = None
            

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
        
    
    def get_prediciton_positions(self):
        if self.S_pred is None:
            self.find_prediction_positions()
        return self.S_pred
    
    
    def combine_samples_and_predictions(self, A_smp, A_pred, ret_3d = True):
        # Dimension along z-axis
        M = A_smp.shape[0]
        # Base
        A_hat_2d = np.zeros((M, self.Nfull))
        # Assign
        A_hat_2d[:, self.idx_smp] = np.copy(A_smp)
        A_hat_2d[:, self.idx_pred] = np.copy(A_pred)
        
        if ret_3d == True:
            return np.reshape(A_hat_2d, (M, self.N_batch, self.N_batch), 'F') # shape = M x N_batch x N_batch
        else:   
            return A_hat_2d # shape = M x N_batch**2
        

class InverseDistanceWeighting():
    
    def __init__(self, S_smp, A_smp_2d, maxlag, add_fluctuation = True, precision = 10):
        """
        Parameters
        ----------
        
        """
        # Unfold
        self.S_smp = np.copy(S_smp) # Scan positions in [m]
        self.A_smp = np.copy(A_smp_2d)
        self.precision = np.copy(precision)
        
        # Maxlag
        self.maxlag = np.copy(maxlag)
        if add_fluctuation == True:
            self.maxlag += 10.0**(- self.precision)
            
        # Base class for batch interpolation
        self.batchitp = None
    
    
    def find_valid_positions(self, s0):
        # Calculate the lags b/w s0 and sampling spotions
        lag0_smp = np.linalg.norm((self.S_smp - s0), axis = 1) # shape = S_smp.shape[0]
        # Identify the indices of the S_smp within the maxlag
        slh = SpatialLagHandlingScalar(lag0_smp, maxlag = self.maxlag)
        idx_Svalid = slh.get_valid_lagindices()
        return idx_Svalid
        
        
        
    def predict_pointwise(self, s0):
        # (1) Find the valid positions
        idx_Svalid = self.find_valid_positions(s0) # shape = Ns
        
        if len(idx_Svalid) == 0:
            #print('IDW: No valid point found for s0 = {}'.format(s0))
            a0hat = np.zeros(self.A_smp.shape[0])
        
        else:
            S_valid = self.S_smp[idx_Svalid, :] # shape = Ns x 2
            A_valid = self.A_smp[:, idx_Svalid] # shape = M x Ns
            # (2) Calcuate the valid lags
            lags0_valid = np.linalg.norm((S_valid - s0), axis = 1) # shape = Ns
            # (3) Compute weights: antiproportional to the distance
            w0 = 1/lags0_valid # shape = Ns
            w0 = w0/np.sum(w0)
            # (4) Predict 
            a0hat = np.dot(A_valid, w0) # shape = M
        
        return a0hat
    
    
    def predict_blockwise(self, N_batch, grid_spacing, ret_3d = True):
        # Initialize the batch inteprolation class
        self.batchitp = BatchInterpolation3D(self.S_smp, N_batch, grid_spacing, self.maxlag)
        # Find the prediction positions
        S_pred = self.batchitp.get_prediciton_positions() # shape = Nc x 2, in [m]
        # Iterate over each position in S_pred
        A_pred = np.apply_along_axis(self.predict_pointwise, 1, S_pred).T # shape = M x Nc
        # Compbine samples & prediction
        A_hat = self.batchitp.combine_samples_and_predictions(self.A_smp, A_pred, ret_3d = ret_3d)
        
        return A_hat
        
        
        
def slicewise_spline2d(S_smp, A_smp_2d, N_batch, grid_spacing):
    # Unroll the dimension along z-axis
    M = A_smp_2d.shape[0]
    # Round back the scan positions to the grid points: float in [m] -> int (unitless)
    #P_smp = np.around(S_smp / grid_spacing) # shape = Ns x 2
    
    # Grid points
    x_full = np.around(grid_spacing* np.arange(N_batch), 10)
    y_full = np.copy(x_full)
    
    # Base
    A_hat = np.zeros((M, N_batch, N_batch))
    
    for sliceNo in range(M):
        # Compute spline function
        slice_smp = A_smp_2d[sliceNo, :]
        f = interp2d(S_smp[:, 0], S_smp[:, 1], slice_smp, kind = 'cubic')
        # Interpolate
        slice_pred = f(x_full, y_full) # shape = N_batch x N_batch
        # Assign
        A_hat[sliceNo, :, :] = slice_pred.T
        
    return A_hat
        
        
        
def moving_window_averaging(data_batch, x, y, N_batch, Nx, Ny):
    N = int(N_batch/2)
    
    # Base matrices
    X1 = np.ones((N, N))
    X2 = round(1/2, 3)* np.ones((N, N))
    X3 = round(1/4, 3)* np.ones((N, N))
    
    def mutiplication_matrix(a, b, c, d):
        """ 
            arr = [[a, b]
                   [c, d]]
        """
        upper = np.concatenate((a, b)) # shape = 2N x N
        bottom = np.concatenate((c, d)) # shape = 2N x N
        return np.concatenate((upper, bottom), axis = 1) # shape 2N x 2N
    
    # Construct multiplication matrix
    # (1)
    if x == 0 and y == 0:
        arr = mutiplication_matrix(X1, X2, X2, X3)
        
    # (2)
    elif y == 0 and x >= N and x < (Nx - N_batch):
        arr = mutiplication_matrix(X2, X2, X3, X3)
        
    # (3)
    elif x == (Nx - N_batch) and y == 0:
        arr = mutiplication_matrix(X2, X1, X3, X2)
    
    # (4)    
    elif x == 0 and y >= N and y < (Ny - N_batch):
        arr = mutiplication_matrix(X2, X3, X2, X3)
    
    # (6)
    elif x == (Nx - N_batch) and y >= N and y < (Ny - N_batch):
        arr = mutiplication_matrix(X3, X2, X3, X2)
        
    # (7)
    elif x == 0 and y == (Ny - N_batch):
        arr = mutiplication_matrix(X2, X3, X1, X2)
        
    # (8)
    elif y == (Ny - N_batch) and x >= N and x < (Nx - N_batch):
        arr = mutiplication_matrix(X3, X3, X2, X2)
        
    # (9)
    elif x == (Nx - N_batch) and y == (Ny - N_batch):
        arr = mutiplication_matrix(X3, X2, X2, X1)
    
    # (5)
    else:
        arr = mutiplication_matrix(X3, X3, X3, X3)
    
    # Scale
    data_scaled = data_batch* arr[None, ...]
    return data_scaled    
        
        
#%%
if __name__ == '__main__':
    Nx = 25
    Ny = 20
    M = 5
    N_batch = 10
    
    p_batch_all = int(N_batch/2)* get_all_grid_points(int((Nx - N_batch)/5) + 1, int((Ny - N_batch)/5) + 1) 
    
    tens1 = np.ones((M, Nx, Ny))
    tens2 = np.zeros(tens1.shape)
    
    for batchNo, p in enumerate(p_batch_all):
        print('=================')
        print('p = {}'.format(p))
        # Assign
        x_start, y_start = p
        
        arr = moving_window_averaging(tens1[:, x_start : x_start + N_batch, y_start : y_start + N_batch], 
                                      x_start, y_start, N_batch, Nx, Ny)
        tens2[:, x_start : x_start + N_batch, y_start : y_start + N_batch]  += np.copy(arr)
        
    
    
        
        
    
        