# -*- coding: utf-8 -*-
import numpy as np
import abc
import scipy.signal

from ..utils.parameter_parsers import ParameterParser
from ..definitions import units
ureg = units.ureg
"""
#============ Tools for Pulse Forming ============#

"""

class PulseFormer(abc.ABC):
    
    def __init__(self):
        r""" Constructor
        """
        super().__init__()
        # Variable setting
        self.pulse_ip = None # in-phase component 
        self.pulse_q = None # quadrature component 
        self.env = None # envelope

    @abc.abstractmethod
    def pulse_unit_handling(self):
        r""" Unit handling in case where the parameters are with unit (i.e. ureg)
        """
        
    @abc.abstractmethod
    def compute_pulse(self):
        r""" Compute pulse
        """
    
    def get_pulse(self, retquad = False, retenv = False, *args):
        r""" Return the calculated pulse
        Parasmeters
        -----------
            retquad : boolaen, optional
                If True, return the quadrature (imaginary) as well as the real part
                of the signal.  Default is False.
            retenv : boolaen, optional
                If True, return the envelope of the signal as well.  Default is False.
        
        Retruns
        -------
            self.pulse_ip : np.array (without unit)
                The in-phase (real) component of the Gaussian pulse
            self.pulse_q : np.array (without unit), optional
                The quadrature component of the Gaussian pulse, returned only if retquad is True
            self.env : np.array (without unit), optional
                The envelope of the Gaussian pulse, returned only if retenv is True
        """
        self.compute_pulse()
        if retquad == False and retenv == False:
            return self.pulse_ip
        if retquad == True and retenv == False:
            return self.pulse_ip, self.pulse_q
        if retquad == False and retenv == True:
            return self.pulse_ip, self.env
        if retquad == True and retenv == True:
            return self.pulse_ip, self.pulse_q, self.env
        
        
    def convolve_pulse_and_reflectivity(self, ref_vec, pulse, Nt, *args):
        r""" Convolve a reflectivity vector and a pulse (either in-phase or quadrature)
        
        Parameters
        ----------
            ref_vec : np.array with the size of Ntdict x 1
                containing few 1s and elsev is 0, a column vector of self.reflectivity_matrix
            pulse : np.array 
                either in-phase (cos) or quardature (sin) component
            Nt : int
                number of time domain samples (w.r.t. ROI)
        
        Returns
        -------
            shifted_pulse : np.ndarray with the size of Ntdict x 1
        """
        # For adjusting the length of the output signal to the desired length, i.e. Ntdict
        upper_limit = int(len(pulse)/2)
        lower_limit = int(upper_limit + Nt)
        # Convolution
        y = np.convolve(pulse, ref_vec)
        # Adjust length
        shifted_pulse = y[range(upper_limit, lower_limit)]
        
        return shifted_pulse


class PulseFormerScipyGausspulse(PulseFormer):
    r""" Pulse forming using scipy.signal.gausspulse
    This allows to form only the portion of the pulse according to the desired length[S] of the pulse
    
    Example usage:
        (1) pulseformer = PulseFormerScipyGausspulse(pulse_parmws, with_unit = True)
        (2) pulseformer.compute_pulse()
        (3) pulse = pulseformer.get_pulse() # using the in-phase component as a pulse
        (4) reflectivity = np.zeros(Ntdict)
            reflectivity[tof_idx] = 1
        (5) ascan = pulseformer.convolve_pulse_and_reflectivity(reflectivity, pulse)
    """
    
    def __init__(self, pulse_params, with_unit):
        r""" Constructor
        Parmameters
        -----------
            pulse_params : dict (with/without unit)
                Parameter dictionary containing 'tPulse', 'fCarrier', 'B', 'fS'
            with_unit : boolean
                True, if pulse parameters are with unit
        """
        super().__init__()
        self.pulse_params = dict(pulse_params)
        self.with_unit = with_unit
        # Set variables used in this class
        self.tPulse = None # length of the pulse [S]
        self.fCarrier = None # carrier frequency [Hz]
        self.B = None # relative bandwith, unitless
        self.fS = None # sampling frequency [Hz]
        self.alpha = None # Gaussian window width controlling factor
               
        
    def pulse_unit_handling(self, *args):
        r""" Unit handling in case where the parameters are with unit (i.e. ureg)
        """
        # call parameter parser class
        prampars = ParameterParser()
        # set required parameters and their type
        prampars.define_required_parameters(['tPulse', 'fCarrier', 'B', 'fS'])
        prampars.define_parameter_types({'tPulse': {'type': {'[time]': 1.0},
                                                    'range': {'Min': 0, 'Max': 10e-6}},
                                         'fCarrier': {'type': {'[time]': -1.0},
                                                     'range': {'Min': 0, 'Max': 100e6}},
                                         'B': {'type': None,  # for scipy.signal, B needs to be relative to fCarrier!!!
                                                      'range': {'Min': 0, 'Max': 2}},
                                         'fS': {'type': {'[time]': -1.0},
                                                      'range': {'Min': 0, 'Max': 100e6}}})
        prampars.set_parameters(self.pulse_params)
        # get parameters
        tPulse = prampars.get_parameter('tPulse')
        fCarrier = prampars.get_parameter('fCarrier')
        B = prampars.get_parameter('B')
        fS = prampars.get_parameter('fS')
        
        return tPulse, fCarrier, B, fS

    def compute_pulse(self, *args):
        r""" Compute pulse using scipy.signal.gausspulse and yield both in-phase and quadrature components
        Pulse calculation should be done WITHOUT unit! 
        
        """
        # Parameter handling
        if self.with_unit == True: 
            self.tPulse, self.fCarrier, self.B, self.fS = self.pulse_unit_handling()
        else:
            self.tPulse = self.pulse_params['tPulse']
            self.fCarrier = self.pulse_params['fCarrier']
            self.B = self.pulse_params['B']
            self.fS = self.pulse_params['fS']
        # calculate pulse length in samples
        Npulse = self.tPulse * self.fS
        # time vector
        if np.mod(Npulse, 2) == 0:
            t_vec = (np.linspace(0, Npulse, Npulse, endpoint=False) - Npulse / 2) / self.fS
        else:
            t_vec = (np.linspace(0, Npulse-1, Npulse, endpoint=True) - (Npulse-1) / 2) / self.fS
        # Get pulse
        self.pulse_ip, self.pulse_q, self.env = scipy.signal.gausspulse(t_vec, self.fCarrier, self.B, 
                                                                        retquad=True, retenv = True)
        # Window width controlling factor alpha 
        # necessary parameters obtained from scipy
        bwr = -6 #from scipy.siganl.gausspulse
        ref = pow(10.0, bwr / 20.0)
        # divide with fS**2 to compensate fCarrier**2...?
        self.alpha = -(np.pi * self.fCarrier * self.B) ** 2 / (4.0 * np.log(ref)) # unitless!
        
        
class PulseFormerGabor(PulseFormer):
    r""" Pulse forming by calculating the Gabor pulse directly. 
    In this class, the time shift by ToF is taken into account, meaning the returned pulse is identical to the 
    corresponding A-Scan.
    
    Example usage:
        (1) pulseformer = PulseFormerGabor(pulse_params, with_unit = True)
        (2) ascan = pulseformer.get_pulse() # only the in-phase component (i.e. real part) is considered
    """
    
    def __init__(self, pulse_params, tof, with_unit):
        r""" Constructor
        Parmameters
        -----------
            pulse_params : dict (with/without unit)
                Parameter dictionary containing 'fCarrier', 'Nt', 'fS', 'alpha'
            tof : float, in [S], without unit!!!
                The time of flight
            with_unit : boolean
                True, if pulse parameters are with unit
        """
        super().__init__()
        self.pulse_params = dict(pulse_params)
        self.with_unit = with_unit
        # Set variables used in this class
        self.fCarrier = None # carrier frequency [Hz]
        self.Nt = None # number of time domain samples w.r.t. ROI
        self.fS = None # sampling frequency [Hz]
        self.tof = tof # time of flight [S]
        self.alpha = None # Gaussian window width controlling factor [Hz**2]
        
               
        
    def pulse_unit_handling(self, *args):
        r""" Unit handling in case where the parameters are with unit (i.e. ureg)
        """
        # call parameter parser class
        prampars = ParameterParser()
        # set required parameters and their type
        prampars.define_required_parameters(['fCarrier', 'Nt', 'fS', 'alpha'])
        prampars.define_parameter_types({'fCarrier': {'type': {'[time]': -1.0},
                                                     'range': {'Min': 0, 'Max': 100e6}},
                                         'Nt': {'type': None},
                                         'fS': {'type': {'[time]': -1.0},
                                                      'range': {'Min': 0, 'Max': 100e6}},
                                         'alpha': {'type': {'[time]': -2.0},
                                                      'range': {'Min': 0, 'Max': 100e12}}},
                                        )
        prampars.set_parameters(self.pulse_params)
        # get parameters
        fCarrier = prampars.get_parameter('fCarrier')
        Nt = prampars.get_parameter('Nt')
        fS = prampars.get_parameter('fS')
        alpha = prampars.get_parameter('alpha')
        
        return fCarrier, Nt, fS, alpha
    

    def compute_pulse(self, *args):
        r""" Compute pulse by calculaing Gabor function directly.
        THe obtained pulsse and envelope have the SAME length as self.Nt => no convolution required!
        
        """
        # Parameter handling
        if self.with_unit == True: 
            self.fCarrier, self.Nt, self.fS, self.alpha = self.pulse_unit_handling()
        else:
            self.fCarrier = self.pulse_params['fCarrier']
            self.Nt = self.pulse_params['Nt']
            self.fS = self.pulse_params['fS']
            self.alpha = self.pulse_params['alpha']
        # Set t, omega
        self.t = (np.arange(0, self.Nt)/ self.fS - self.tof) # reflecting the time shift by ToF
        omega = 2* np.pi* self.fCarrier
        # Calculate the Gaussian window (i.e. the envelope of the pulse)
        self.env = np.exp(-self.alpha* self.t**2)
        # Calculate the pulse
        self.pulse_ip = self.env* np.cos(omega* self.t)
        self.pulse_q = self.env* np.sin(omega* self.t)
        
    







