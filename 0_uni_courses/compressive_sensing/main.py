r"""
Compressive Sensing HW
"""
import numpy as np
import matplotlib.pyplot as plt

r"""
Measurement data model:
    s = H* a
        s: size= N x 1
            Measurement data (A-Scan), time domain
        H: size = N x N
            Measurement dictionary, mapping spatial domain -> time domain
        a: size = N x 1
            Reflectivity, sparse vectr to recover, spatial domain
            norm(a)_0 = K
Compressed measurement data:
    b = Phi* s = Phi* H* a
        b: size= M x 1
            Compressed measurement data, frequency domain
        Phi: size = M x N
            Measuremet kernel, row-subselected Fourier matrix, mapping time domain -> frequency domain
            (e.g.) M = 4 -> f = [f1, f2, f3, f4]
                Phi = np.array([e^(-j*2pi*f1*t), e^(-j*2pi*f2*t), e^(-j*2pi*f3*t), e^(-j*2pi*f4*t)]).T
"""
plt.close('all')
#======================================================================================================= Functions ====#
class DataCompression():
    def __init__(self, params):
        # Global parmeter setting
        self.N = params['N']
        self.fS = params['fS']
        self.fC = params['fC']
        self.B = params['B']
        self.dt = 1/self.fS
        self.t = self.dt* np.arange(0, N)
        self.p_t = None # Base pulse
        self.L = None # Pulse length
        self.H = None # Measurement dictionary
        self.Phi = None # Measurement kernel
        self.PhiH = None # Product of Phi and H for easy access
        
    def calculate_pulse(self):
        t_pulse = np.arange(-1, 1, self.dt) # Range: from -2pi to 2pi
        self.p_t = np.exp(-(t_pulse**2)/B)* np.cos(2*np.pi*fC*t_pulse) # Base pulse 
        self.L = len(self.p_t)
    
    def convolve_pulse(self, vec):
        h = np.convolve(self.p_t, vec) # size = N + L
        h = h[int(self.L/2):-int(self.L/2)+1] # size = N
        return h
    
    def Fourier_coefficients(self, f):
        return np.exp(-1j*2*np.pi*f*self.t)
        
    def normalize_column_vector(self, vec):
        # Normalize the vector, s.t. norm(PhiH[:, col]) == 1
        norm = np.linalg.norm(vec)
        return vec/norm
        
    def compute_dictionary(self):
        self.calculate_pulse()
        # Column vector = covolution b/w the base pulse and the time shifted dirac delta
        self.H = np.apply_along_axis(self.convolve_pulse, 0, (1 + 0j)*np.identity(self.N))
        
        
    def compute_kernel(self, M):
        # Random selection of the kernel components
        mu = self.fC/self.fS
        sigma = 1/np.sqrt(2*self.B) # Fit the window width to the Gaussian window width of the pulse
        f = fS*np.random.normal(mu, sigma, size = M) # Select K random freq. components
        # Kernel = row-subselected Fourier matrix
        self.Phi = np.apply_along_axis(self.Fourier_coefficients, 0, f[np.newaxis, :]).T # Size = K x N
        self.PhiH = np.dot(self.Phi, self.H)
        # Normalize each column vector s.t. norm(PhiH[:, n]) == 1
        self.PhiH = np.apply_along_axis(self.normalize_column_vector, 0, self.PhiH)
    
        
    def get_dictionary(self):
        return self.H
    
    def get_measurement_data(self, a):
        self.compute_dictionary()
        return np.dot(self.H, a)           
    
    def get_PhiH(self):
        return self.PhiH
    
    def get_compressed_data(self, M):
        self.compute_kernel(M)
        return np.dot(self.PhiH, a)


def Hermitian(arr):
    return np.conj(arr).T    

    
def reconstruction(b, PhiH, target, OMP):
    r"""
    Parameters
    ----------
        b: np.array(K)
            Compressed measurement data
        PhiH: np.array(K, N)
            Product of the measurement kernel, PHi, and the measurement dictionary H
         target: float
             Breack condition, when the norm(residual) <= target
        OMP: boolean
            True, if OMP should be performed
            False, if MP should be performed
    """
    # Initial setting
    r = np.copy(b) # Residual, i.e. error b/w the compressed measurement data, b, and our model, b_hat = PhiH* a_hat
    error = np.linalg.norm(r)
    error_list = []
    S = [] # For OMP: set to store the K largest component (K = # of non-zero elements)
    a_hat = np.zeros(N, dtype = complex)
    n_iter = 0 # for MP
    while error > target:
        c = np.dot(Hermitian(PhiH), r) # Correlation
        idx = np.argmax(c) 
        # OMP
        if OMP == True:
            S.append(idx) # Update the set
            PhiH_S = PhiH[:, S] # Size = K x idx
            a_S = np.dot(np.linalg.pinv(PhiH_S), b) # Lesat square solution
            a_hat[S] = a_S # Update the estimation
            r = b - np.dot(PhiH_S, a_S) # Update the residual            
        # MP
        else:
            a_hat[idx] += c[idx] # Update the estimation
            r = r - c[idx]* PhiH[:, idx] # Update the residual
            n_iter += 1
            print(idx)
            if n_iter > 100:
                break
        # Current error    
        error = np.linalg.norm(r)
        error_list.append(error)
    return a_hat, error_list
    
#=============================================================================== Parameters, pulse & sparse vector ====#
N = 220 # dimension (both in spatial and temporal domain)
fS = 100 # Hz
fC = 20 # Hz
B = 1/100 #Hz**2
# Sparse vector
peak_idx = np.array([50, 130]) # Position of the peaks, spatial domain
K = len(peak_idx) # l0 norm of a
a = np.zeros(N, dtype = complex) # Sparse vector to recover
a[peak_idx] = 1 + 0j

# Compression
M = 25*K # Dimenion of compressed data -> 11*K works for OMP

#================================================================================== Un/compressed measurement data ====#
dc = DataCompression({'N': N, 'fS': fS, 'fC': fC, 'B': B})
# Uncompressed data generation
dc.compute_dictionary()
H = dc.get_dictionary() # Measurement dictionary
s_t = dc.get_measurement_data(a)# Actual measurement data (A-Scan), size = N x 1
# Data compression
b = dc.get_compressed_data(M) # Compressed data ,size = M x 1
PhiH = dc.get_PhiH() # = np.dot(Phi, H), noemalized

#================================================================================================== Reconstruction ====#
target = 10**-3
# MP
ahat_MP, error_MP_list = reconstruction(b, PhiH, target, False) 
# OMP
ahat_OMP, error_OMP_list = reconstruction(b, PhiH, target, True)
# Evaluation
error_MP = np.array(error_MP_list)
error_OMP = np.zeros(error_MP.shape)
error_OMP[:len(error_OMP_list)] = error_OMP_list
#=========================================================================================================== Plots ====#
plt.figure(1)
plt.plot(np.abs(a), label = 'Reflector')
plt.plot(np.abs(ahat_MP), label = 'Reconstruction')
plt.title('a_hat MP')
plt.xlabel('z/dz')
plt.ylabel('Amplitude')
plt.legend()


plt.figure(2)
plt.plot(np.abs(a), label = 'Reflector')
plt.plot(np.abs(ahat_OMP), label = 'Reconstruction')
plt.title('a_hat OMP')
plt.xlabel('z/dz')
plt.ylabel('Amplitude')


plt.figure(3)
plt.plot(error_MP, label = 'MP')
plt.plot(error_OMP, label = 'OMP')
plt.title('Error at each iteration step')
plt.xlabel('Iterations')
plt.ylabel('norm(r)_2')
plt.legend()



