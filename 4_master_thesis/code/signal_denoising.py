# -*- coding: utf-8 -*-
"""
Signal Denoising
"""
import numpy as np
import numpy.fft as fft


def denoise_svd(data, d = None, sigma_n = None):
    r""" Denoise a noisy multi-dimensional data set using SVD
    
    Parameters
    ----------
        data: np.ndarray, ndim >= 2 (i.e. 2D, 3D, ...)
            Noisy data with
                ndim = m
                shape = N1 x N2 x N3 x ... x Nm-1 x Nm
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
    # U.shape = N1 x N2 x N3 ... x Nm-1 x Nm-1, ndim = m
    # Vh.shape = N1 x N2 x N3 ... x Nm x Nm, ndim = m
    # S.shape = N1 x N2 x N3 ... x min(Nm-1, Nm), ndim = m-1 
    U, S, Vh = np.linalg.svd(data)
    # Define the rank according to the noise energy
    if d is None:
        if sigma_n is None:
            raise AttributeError('denoise_svd: provide either rank or the noise power')
        else:
            d = len(np.argwhere(S > np.sqrt(sigma_n)))
    # economy size SVD
    if d != 1:
        Us = U[..., :, :d] # shape = N1 x N2 x N3 ... x Nm-1 x d
        Ss = S[..., None, :d] # shape = N1 x N2 x N3 ... x None x d
        Vhs = Vh[..., :d, :] # shape = N1 x N2 x N3 ... x d x Nm
        return np.matmul(Us * Ss, Vhs)
    else:
        Us = U[:, 0].reshape(U.shape[0], 1)
        Ss = S[:d]
        Vhs = Vh[0, :].reshape(1, Vh.shape[1])
        return Ss* np.dot(Us, Vhs)


def denoise_wavelet():
    pass


def denoise_lowpass(s_t, f_cutoff, real_signal = True, ret_spectrum = False):
    """ Signal denoising via low-pass

    Parameters
    ----------
    s_t : np.ndarray
        Signal to be denoised, in time domain
        The 0-th axis should correspond to the time axis (= the axis to compute FFT)
    f_cutoff : int (unitless!!!!!)
        Freq. bin(= bin index) corresponds to the cutoff frequency 
    real_signal : boolean, optional (True by default)
        True, if the given signal is real-valued
    ret_spectrum : boolean, optional (False by default)
        True, if the low-passed spectrum is to be returned

    Returns
    -------
    s_t_clean: np.ndarray
        Low-passed signal 
    S_f_clean : np.ndarray, complex
        Low-passed freq. spectrum (returned only if ret_spectrum == True)
    """
    M = s_t.shape[0] # Signal lenghth along time-axis
    
    # Real-valued signal
    if real_signal == True:
        S_f_noisy = fft.rfft(s_t, axis = 0) 
        S_f_clean = 1j* np.zeros(S_f_noisy.shape)
        S_f_clean[:f_cutoff, :, :] = np.copy(S_f_noisy[:f_cutoff, :, :])
        s_t_clean = fft.irfft(S_f_clean, M, axis = 0)
    # Complex-valued signal    
    else:
        S_f_noisy = fft.fft(s_t, axis = 0)
        S_f_clean = 1j* np.zeros(S_f_noisy.shape)
        S_f_clean[:f_cutoff, :, :] = np.copy(S_f_noisy[:f_cutoff, :, :]) # Positive freq.
        S_f_clean[-f_cutoff:, :, :] = np.copy(S_f_noisy[-f_cutoff:, :, :]) # Negative freq.
        s_t_clean = fft.ifft(S_f_clean, M, axis = 0)
    
    if ret_spectrum == False:
        return s_t_clean
    else:
        return s_t_clean, S_f_clean
        
    
    
