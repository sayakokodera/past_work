# -*- coding: utf-8 -*-
"""
Signal Denoising
"""
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

def denoise_svd(data, d = None, sigma_n = None):
    r""" Denoise a noisy data set using SVD
    
    Parameters
    ----------
        data: np.ndarray
            Noisy data
        d: int (None by default)
            Dimension of the signal space 
        sigma_n: float (None by default)
            Noise power to determine the 
        => either d or sigma_n should be provided
        
    Returns
    -------
        Rank-d approximation of the provided noisy data
    """
    # SVD
    U, S, Vh = linalg.svd(data)
    # Define the rank according to the noise energy
    if d is None:
        if sigma_n is None:
            raise AttributeError('denoise_svd: provide either rank or the noise power')
        else:
            d = len(np.argwhere(S > np.sqrt(sigma_n)))
    # economy size SVD
    if d != 1:
        Us = U[:, :d]
        Ss = S[:d]* np.identity(d)
        Vhs = Vh[:d, :]
        return np.dot(np.dot(Us, Ss), Vhs)
    else:
        Us = np.array([U[:, 0]]).T
        Ss = S[:d]
        Vhs = np.array([Vh[0, :]])
        return Ss* np.dot(Us, Vhs)


def denoise_wavelet():
    pass

