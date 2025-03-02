# -*- coding: utf-8 -*-
"""
 FISTA
"""
import numpy as np
import time
import scipy.sparse.linalg as sclin

from display_time import display_time

class FISTA():
    r""" Find an optimal solution x for A*x \approx b, where
        A: known (ill-conditioned) matrix (n x m)
        x: unknown sparse vector (m x 1)
        b: known measurement vector (n x 1)
        
    Cost fct: min_{x} F(x)
        where   F(x) = f(x) + g(x)
        with    f(x) = \vert A*x - b \vert_{2}^{2} 
                g(x) = lambda* \vert x \vert_{1}
        
    2 steps in FISTA:
        (1) Gradient descent in f(x) -> fidelty
        (2) Soft thresholding considering g(x) -> sparsity 
    """
    
    def __init__(self, Lambda, Niteration):
        r""" Constructor
        Parameters:
        -----------
            Lambda: float 
                Sparsity parameter
            Niteration: int 
                Number of iteration
        """
        self.Lambda = np.copy(Lambda)
        self.Niteration = np.copy(Niteration)
        self.dot_AHA = None # = np.dot(A^{H}, A)
        self.dot_AHb = None # = np.dot(A^{H}, b)
        self.L = None # smallest Lipschitz constant
        self.threshold = None # threshold value, should be >0 !!!!!
        self.complex_values = False # True, if A and/or b are complex
        self.x = None # Solution
        
    
    def gradient_descent(self, y):
        gradf = (np.dot(self.dot_AHA, y) - self.dot_AHb)
        if self.complex_values == False:
            gradf = 2* gradf
        return y - gradf / self.L
    
    
    def soft_thresholding(self, y):
        y_thresholded = np.maximum(np.abs(y) - self.threshold, 0)
        x = np.sign(y)* y_thresholded # Keep the sign
        return x
    
    
    def largest_eigenvalue(self, A, svd_params):
        r""" Calculate the smallest Lipschitz constant of the gradf, which is the largest eigenvalue of
        A^{H}* A
        """
        # Exact SVD calculation
        if svd_params is None:
            L = np.linalg.svd(A, compute_uv = False)[0]**2
        # SVD  approximation for faster computation
        else:
            L = sclin.svds(A, tol = svd_params['tol'], maxiter = svd_params['tol'], 
                           return_singular_vectors = False)[0]**2
        return L
    
    
    def compute(self, A, b, complex_values = False, L = None, svd_params = None):
        r"""
        Parameters
        ----------
            complex_values : boolean (False by default)
                To specify whether the Hermitian transpose of A or the plain transpose should be used
            L : float (None by default)
                Stepsize for FISTA (ideally largest eigenvalue of A^{H}*A)
                If it is None, then L is calculated in this class
            svd_params : dict (None by default)
                When L should be calculated in this class, and the matrix A is large, SVD can be very time consuing.
                In such case, approximation of SVD is computed (w/ sclin.svds) and the parameters for this function
                can be specified in this dictionary.
                (e.g.) tol, maxiter
        """
        # Calculate the dot products
        print('Dot product')
        start = time.time()
        if complex_values == False:
            self.dot_AHA = np.dot(A.T, A) # = A^{T}* A
            self.dot_AHb = np.dot(A.T, b) # = A^{T}* b
        else:
            self.complex_values == True
            self.dot_AHA = np.dot(np.conj(A).T, A) # = A^{H}* A
            self.dot_AHb = np.dot(np.conj(A).T, b) # = A^{H}* b
        display_time(np.around(time.time() - start, 3))
            
        # Calculate the stepsize (= largest eigenvalue of dot_AHA = latgest singularvalue(of A) **2)
        start = time.time()
        if L is not None:
            self.L = np.copy(L)
        else:
            self.L = self.largest_eigenvalue(A, svd_params)
            
        # Calculate the threshold value
        self.threshold = 0.5* self.Lambda/self.L
        
        # Initial setting
        if self.complex_values == False:
            x = np.zeros(A.shape[1])
        else:
            x = np.zeros(A.shape[1], dtype = complex)
        tnew = 1
        ynew = np.copy(x)
        
        # Iteration
        for k in range(self.Niteration):
            start = time.time()
            
            xold = np.copy(x)
            t = np.copy(tnew)
            y = np.copy(ynew)
            
            # Gradient descent
            y_gd = self.gradient_descent(y)
            
            # Soft thresholding
            x = self.soft_thresholding(y_gd)
            
            # Update t & y
            tnew = 0.5* (1 + np.sqrt(1 + 4* (t**2)))
            ynew = x + ((t - 1)/tnew)* (x - xold)
        
        # Solution after Niteration
        self.x = np.copy(x)
 
       
    def get_solution(self):
        return self.x
        
        
    
    
    
    
    
        
        
        
    
        
