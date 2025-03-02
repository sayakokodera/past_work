# -*- coding: utf-8 -*-
"""
Parameter estimation of SmartInspect measurement data

Created on Wed Jun 17 10:56:15 2020

@author: sako5821
"""
import numpy as np
import matplotlib.pyplot as plt
from pulse_former import PulseFormerGabor
import scipy.io as scio
import scipy.signal as scsi
import abc

from ultrasonic_imaging_python.utils.file_readers import SmartInspectReader
from signal_denoising import denoise_svd


plt.close('all')

#%% Functions 
class ParameterEsimation(abc.ABC):
    def __init__(self, data, fC, fS, Nt_offset, Nt, with_envelope = False):
        self.data = np.copy(data)
        self.fC = fC
        self.fS = fS
        self.Nt_offset = Nt_offset
        self.with_envelope = with_envelope
        self.Nt = Nt
        self.alpha = None # Bandwidth factor
        self.phi = None # Phase
        self.r = None # Chirp rate
        self.e_all = None
        

    @abc.abstractmethod
    def set_variable_values(self):
        r"""
        """

    def data_model(self):
        # Find the peak position
        # Case: data = A-Scan
        if self.with_envelope == False:
            z_peak_idx = np.argmax(np.abs(scsi.hilbert(self.data)))
        # Case: data = envelope
        else:
            z_peak_idx = np.argmax(np.abs(self.data))
        tau = (self.Nt_offset + z_peak_idx)/self.fS # [S] - corresponds to the peak position of the pulse
        # Compute the synthetic pulse
        pformer = PulseFormerGabor(self.Nt, self.fS, self.fC, self.alpha, self.r)
        if self.with_envelope == False:
            pformer.calculate_ip_pulse(tau, self.phi)
            model = pformer.get_ip_pulse(self.Nt_offset)
        else:
            pformer.calculate_envelope(tau)
            model = pformer.get_envelop(self.Nt_offset)
        return model

    def model_vs_data(self):
        model = self.data_model()
        err = np.linalg.norm(self.data - model)
        return err
    
    @abc.abstractclassmethod
    def calculate_error(self):
        r"""
        """
    
    def get_error_all(self):
        return self.e_all
    
    def optimize_parameter(self, param_values):
        self.e_all = np.apply_along_axis(self.calculate_error, 0, np.array([param_values]))
        return param_values[np.argmin(self.e_all)]
  
    
class ParameterEstimationPhase(ParameterEsimation):
    def __init__(self, data, fC, fS, Nt_offset, Nt, with_envelope = False):
        super().__init__(data, fC, fS, Nt_offset, Nt, with_envelope)
        
    def set_variable_values(self, alpha, r):
        self.alpha = np.copy(alpha)
        self.r = np.copy(r)

    def calculate_error(self, phi):
        self.phi = phi
        err = self.model_vs_data()
        return err
    
class ParameterEstimationBandwidthFactor(ParameterEsimation):
    def __init__(self, data, fC, fS, Nt_offset, Nt, with_envelope = False):
        super().__init__(data, fC, fS, Nt_offset, Nt, with_envelope)
        
    def set_variable_values(self, phi, r):
        self.phi = np.copy(phi)
        self.r = np.copy(r)

    def calculate_error(self, alpha):
        self.alpha = alpha
        err = self.model_vs_data()
        return err
    

class ParameterEstimationChirpRate(ParameterEsimation):
    def __init__(self, data, fC, fS, Nt_offset, Nt, with_envelope = False):
        super().__init__(data, fC, fS, Nt_offset, Nt, with_envelope)
        
    def set_variable_values(self, alpha, phi):
        self.alpha = np.copy(alpha)
        self.phi = np.copy(phi)

    def calculate_error(self, r):
        self.r = r
        err = self.model_vs_data()
        return err    


def plot_data(data, fig_no, title):
    plt.figure(fig_no)
    plt.imshow(data)
    plt.title(title) 
    

def load_txt(fname):
    with open(fname) as f:
        lines = (line for line in f)
        data = np.loadtxt(lines, skiprows=1)
    return data


def load_raw_si_data(path):
    si_reader = SmartInspectReader(path)
    data_all = si_reader.get_all()
    return data_all['data']


#%% Parameter Estimation 
path = 'SmartInspect_data/200708/calibration'
## MUSE
#data_all = scio.loadmat('MUSE/measurement_data.mat')
#data = data_all['data']
## SI A-Scans
#data = np.load('npy_data/SmartInspect/200623/A_aa_25pixels.npy')
#cscan = np.max(np.abs(data), axis = 0)
## SI txt data
#data_noisy = load_txt('{}/backwall_echo.txt'.format(path))
## SI back wall echoes
#data_noisy = load_raw_si_data(path)
#data_noisy = np.delete(data_noisy, 782, axis = 1)
# npy_data
path = 'npy_data/SmartInspect/200903_2'
fname = '{}/A_bw.npy'.format(path)
data_noisy = np.load(fname)
#data = data[:, :2]

# Info regarding the measurement data & data matrix
fS = 80*10**6 #[Hz] 
tmin = 0 #= 610 In case the raw measurent data from SI is used, specify where to start for a faster computation
tmax = data_noisy.shape[0]
#tmin, tmax = 240, 340
Nx = data_noisy.shape[1] 

# Denoise data
data = denoise_svd(data_noisy, d = 2)

# Variable
fC = 4.48*10**6 #[Hz] 
alpha_base = 8.3*10**12 #[Hz]**2, 8, 8.6
phi_base = 277 # [grad]
r_base = -0.0332 # -1...1
with_envelope = True

# Base
r_opt_all = np.zeros(Nx)
phi_opt_all = np.zeros(Nx)
alpha_opt_all = np.zeros(Nx)

for x in range(Nx):
    # Data in ROI
    env = np.abs(scsi.hilbert(data[tmin:tmax, x])) # for r and alpha
    pulse = np.copy(data[tmin:tmax, x])/env.max() # for phi
    
    # Chirp rate optimization
    r_all = np.around(np.arange(-0.99, 1.0, 0.01), 3)
    pep_r = ParameterEstimationChirpRate(env/env.max(), fC, fS, tmin, tmax, with_envelope = False)
    pep_r.set_variable_values(alpha_base, phi_base)
    r_opt_all[x] = pep_r.optimize_parameter(r_all)

    # alpha optimization
    alpha_all = np.around(np.arange(7.0, 9.0, 0.1), 3)*10**12
    pep_alpha = ParameterEstimationBandwidthFactor(env/env.max(), fC, fS, tmin, tmax, with_envelope = False)
    pep_alpha.set_variable_values(phi_base, r_opt_all[x])
    alpha_opt_all[x] = pep_alpha.optimize_parameter(alpha_all)
    
    # Phase optimization
    phi_all = np.arange(250, 290, 1)
    pep_phi = ParameterEstimationPhase(pulse, fC, fS, tmin, tmax, with_envelope = False)
    pep_phi.set_variable_values(alpha_opt_all[x], r_opt_all[x])
    phi_opt_all[x] = pep_phi.optimize_parameter(phi_all)
    
    del pep_r, pep_alpha, pep_phi
    
    
r_opt = np.mean(r_opt_all)
alpha_opt = np.mean(alpha_opt_all)
phi_opt = np.mean(phi_opt_all)

print('r =  {}'.format(r_opt))
print('phi =  {}'.format(phi_opt))
print('alpha =  {}'.format(alpha_opt))
    
# Test the optimal value
x = 100
env = np.abs(scsi.hilbert(data[tmin:tmax, x]))
a_meas = data[tmin:tmax, x] / env.max()
pep_alpha = ParameterEstimationBandwidthFactor(a_meas, fC, fS, tmin, tmax, with_envelope = False)
pep_alpha.phi = phi_opt
pep_alpha.alpha = alpha_opt
pep_alpha.r = r_opt
a_hat = pep_alpha.data_model()

# Plot
dt = 1/fS
taxis = np.arange(tmin, tmax)* dt

# Pulse: noisy vs denoised data
a_meas_raw = data_noisy[tmin:tmax, x] / np.abs(scsi.hilbert(data_noisy[tmin:tmax, x])).max()
plt.figure(1)
plt.plot(taxis*10**6, a_meas_raw, label = 'a_raw')
plt.plot(taxis*10**6, a_meas, label = 'a_denoise')
plt.title('Raw data vs denoised signal')
plt.legend() 

# Pulse model
plt.figure(2)
plt.plot(taxis*10**6, a_meas_raw, label = 'a_raw')
plt.plot(taxis*10**6, a_hat, label = 'a_hat')
plt.title('Pulse: raw data vs model')
plt.legend() 

# Envelope
plt.figure(3)
plt.plot(taxis*10**6, np.abs(scsi.hilbert(a_meas_raw)), label = 'env_raw')
plt.plot(taxis*10**6, np.abs(scsi.hilbert(a_hat)), label = 'env_hat')
plt.title('Envelope: raw data vs model')
plt.legend() 

# Freq. components
sp_all = np.fft.fft(data_noisy[tmin:tmax, :], 500, axis = 0)
sp_sum = np.sum(sp_all, axis = 1)
freq = np.fft.fftfreq(500)*80
plt.figure(4)
plt.plot(freq, np.abs(sp_sum))
plt.title('Spectrum in freq. domain')


