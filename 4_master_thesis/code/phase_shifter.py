#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 12:49:08 2021

@author: sayakokodera
"""

import numpy as np

class PhaseShifter():
    """ Shifting the phase of a time series signal by a given phase, phi, in freq. domain and transforming back into
    time domain.
    (e.g.):
        * time domain
        x(t) = cos(2* pi* f0* t)
        y(t) = cos(2* pi* f0* t + phi)
        
        * freq. domain
        X(f) = 0.5* ( delta(f - f0) + delta(f + f0) )
        Y(f) = 0.5* ( exp(+1j)* delta(f - f0) + exp(-1j)*delta(f + f0) )
        ==> Y(f>=0) = exp(+1j)*  X(f>=0)
            Y(f<0) = exp(-1j)*  X(f<0)
    """
    def __init__(self):
        self.phi = None
        

    def shift_phase_1d(self, x):
        """ Computing the phase-shift of a 1D signal x in freq. domain, i.e.
            * freq. domain
            Y(f>=0) = exp(+1j)*  X(f>=0)
            Y(f<0) = exp(-1j)*  X(f<0)
            
        Parameters
        ----------
        x : np.array(N), 1D
            Signal to be phase-shifted by phi (time domain)
    
        Returns
        -------
        y: np.array(N), 1D
            Phase-shifted signal (time domain)
            
        Test
        ----
            Check whether the envlope of y is remained identical to that of x
        """
        # Transforom the input signal into freq. domain
        Xf = np.fft.fft(x)
        
        # Phase-shift: f >= 0
        Yf = np.exp(1j* self.phi)* Xf
        # Phase-shift: f < 0
        Yf[int(0.5*len(Xf)):] = np.exp(-1j* self.phi)* Xf[int(0.5*len(Xf)):]
        
        # Back transform into time domain
        y = np.fft.ifft(Yf)
        
        return y
        
    
    def shift_phase(self, x, phi, axis = None):
        """ Shifting the phase of a time series signal by a given phase, phi, in freq. domain and transforming back into
        time domain. This is performed using the function phase_shifter_1d.
        
        Parameters
        ----------
        x : np.array
            Signal to be phase-shifted by phi (time domain)
        phi : float [degree]
            Phase
        axis : int (optional)
            Axis over which FFT is to be performed, should be specified when the input signal is multidimensional
    
        Returns
        -------
        y: np.array (same size as x)
            Phase-shifted signal (time domain)
    
        """
        # Assign the global parameter: phi in [rad]
        self.phi = np.deg2rad(phi) 
        
        # x is 1D
        if x.ndim == 1:
            y = self.shift_phase_1d(x)
        
        # x is multidimensional
        else:
            if axis is None: #2+D, but axis is not specified
                raise AttributeError('phase_shifter: FFT axis is not specified!')
            else:
                y = np.apply_along_axis(self.shift_phase_1d, axis, x)
                
        if np.all(np.isreal(x)) == True:
            y = y.real
        
        return y
    