r"""
Deconvolution
"""
import numpy as np


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
#======================================================================================================= Functions ====#
class DataCompression():
    def __init__(self, params):
        # Global parmeter setting
        self.N = params['N']
        self.fS = params['fS']
        self.fC = params['fC']
        self.B = params['B']
        Nt_offset = params['Nt_offset'] # unitless, index!
        self.t = np.arange(Nt_offset, Nt_offset + self.N)/self.fS
        self.H = None # Measurement dictionary
        self.Phi = None # Measurement kernel
        self.PhiH = None # Product of Phi and H for easy access
        
    
    def fourier_coefficients(self, f):
        return np.exp(-1j*2*np.pi*f*self.t)
        
    def normalize_column_vector(self, vec):
        # Normalize the vector, s.t. norm(PhiH[:, col]) == 1
        norm = np.linalg.norm(vec)
        return vec/norm  
        
    def compute_kernel(self, M, H):
        # Random selection of the kernel components
        mu = self.fC/self.fS
        sigma = 1/np.sqrt(2*self.B) # Fit the window width to the Gaussian window width of the pulse
        f = self.fS *np.random.normal(mu, sigma, size = M) # Select K random freq. components
        # Kernel = row-subselected Fourier matrix
        self.Phi = np.apply_along_axis(self.fourier_coefficients, 0, f[np.newaxis, :]).T # Size = K x N
        # Normalize each column vector s.t. norm(PhiH[:, n]) == 1
        self.Phi = np.apply_along_axis(self.normalize_column_vector, 0, self.Phi)
        # Product of the measurement kernel, PHi, and the measurement dictionary H
        self.PhiH = np.dot(self.Phi, H)
        # Normalize each column vector s.t. norm(PhiH[:, n]) == 1
        self.PhiH = np.apply_along_axis(self.normalize_column_vector, 0, self.PhiH)  
    
    def get_PhiH(self):
        return self.PhiH
    
    def get_compressed_data(self, s, M, H):
        self.compute_kernel(M, H)
        # Normalize the measurement signal
        x = int(0.5* H.shape[1])
        energy = np.linalg.norm(np.dot(self.Phi, H[:, x]))
        s_norm = s/energy
        # Calculate the compressed data
        b = np.dot(self.Phi, s_norm)
        return b


def Hermitian(arr):
    return np.conj(arr).T    

    
def deconvolution(b, PhiH, target, OMP):
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
    a_hat = np.zeros(PhiH.shape[1], dtype = complex)
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
            if n_iter > 100:
                break
        # Current error    
        error = np.linalg.norm(r)
        error_list.append(error)
    return np.abs(a_hat), error_list


class SingleScanFWM():
    def __init__(self, params):
        # Global parmeter setting
        self.N = params['N']
        self.fS = params['fS']
        self.fC = params['fC']
        self.B = params['B']
        Nt_offset = params['Nt_offset']
        self.dt = 1/self.fS
        self.t = self.dt* np.arange(Nt_offset, Nt_offset + N)
        self.p_t = None # Base pulse
        self.L = None # Pulse length
        self.H = None # Measurement dictionary
        
    def calculate_pulse(self):
        t_pulse = np.arange(-1, 1, self.dt) # Range: from -2pi to 2pi
        self.p_t = np.exp(-(t_pulse**2)/B)* np.cos(2*np.pi*fC*t_pulse) # Base pulse 
        self.L = len(self.p_t)
    
    def convolve_pulse(self, vec):
        h = np.convolve(self.p_t, vec) # size = N + L
        h = h[int(self.L/2):-int(self.L/2)+1] # size = N
        return h
    
    def compute_dictionary(self):
        self.calculate_pulse()
        # Column vector = covolution b/w the base pulse and the time shifted dirac delta
        self.H = np.apply_along_axis(self.convolve_pulse, 0, (1 + 0j)*np.identity(self.N))
        
    def get_dictionary(self):
        self.compute_dictionary()
        return self.H   
    

if __name__ == '__main__': 
    import matplotlib.pyplot as plt
    
    plt.close('all')
    #=========================================================================== Parameters, pulse & sparse vector ====#
    N = 220 # dimension (both in spatial and temporal domain)
    fS = 100 # Hz
    fC = 20 # Hz
    B = 1/100 #Hz**2
    # Sparse vector
    peak_idx = np.array([50, 53])#51, 52]) # Position of the peaks, spatial domain
    K = len(peak_idx) # l0 norm of a
    a = np.zeros(N, dtype = complex) # Sparse vector to recover
    #a[peak_idx] = 1/np.sqrt(len(peak_idx)) + 0j
    a[peak_idx[0]] = 1/np.sqrt(2)#0.8
    a[peak_idx[1]] = 1/np.sqrt(2)#0.4
    
    # Compression
    M = 100*K # Dimenion of compressed data -> 11*K works for OMP
    
    #==================================================================================== Measurement data: A-Scan ====#
    fwm = SingleScanFWM({'N': N, 'fS': fS, 'fC': fC, 'B': B, 'Nt_offset' :  500})
    H = fwm.get_dictionary() # Measurement dictionary
    s_t = np.dot(H, a).real# Actual measurement data (A-Scan), size = N x 1
    #============================================================================================ Data compression ====#
    # Data compression
    dc = DataCompression({'N': N, 'fS': fS, 'fC': fC, 'B': B, 'Nt_offset' :  500})
    b = dc.get_compressed_data(s_t, M, H) # Compressed data, size = M x 1
    PhiH = dc.get_PhiH() # = np.dot(Phi, H), noemalized
    
    #============================================================================================== Reconstruction ====#
    target = 10**-3
    # MP
    ahat_MP, error_MP_list = deconvolution(b, PhiH, target, False) #(s_t, H, target, False)#
    # OMP
    ahat_OMP, error_OMP_list = deconvolution(b, PhiH, target, True) #(s_t, H, target, True)#
    # Evaluation
    error_MP = np.array(error_MP_list)
    error_OMP = np.zeros(error_MP.shape)
    error_OMP[:len(error_OMP_list)] = error_OMP_list
    #======================================================================================================= Plots ====#
    plt.figure(1)
    plt.plot(np.abs(a), label = 'Reflector')
    plt.plot(ahat_MP, label = 'Reconstruction')
    plt.title('a_hat MP')
    plt.xlabel('z/dz')
    plt.ylabel('Amplitude')
    plt.legend()
    
    
    plt.figure(2)
    plt.plot(np.abs(a), label = 'Reflector')
    plt.plot(ahat_OMP, label = 'Reconstruction')
    plt.title('Deconvoleved signal (via OMP)')
    plt.xlabel('z/dz')
    plt.ylabel('Amplitude')
    plt.legend()
    
    
    plt.figure(3)
    plt.plot(error_MP, label = 'MP')
    plt.plot(error_OMP, label = 'OMP')
    plt.title('Error at each iteration step')
    plt.xlabel('Iterations')
    plt.ylabel('norm(r)_2')
    plt.legend()
    
    plt.figure(4)
    plt.plot(np.abs(a), label = 'Reflector')
    #plt.plot(np.abs(ahat_OMP), label = 'Reconstruction')
    plt.title('3 consective dirac pulse at z = 50, 51, 52')
    plt.xlabel('z/dz')
    plt.ylabel('Amplitude')
    plt.legend()
    
    plt.figure(5)
    plt.plot(s_t, label = 'A-Scan')
    plt.plot(np.dot(H, ahat_OMP), label = 'Reco')
    plt.title('Original vs reconstructed (OMP) A-Scan')
    plt.xlabel('z/dz')
    plt.ylabel('Amplitude')
    plt.legend()


