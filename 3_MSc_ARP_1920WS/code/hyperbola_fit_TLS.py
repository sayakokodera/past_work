import numpy as np
import scipy.signal as scsi

class HyperbolaFitTLS():
    def __init__(self):
        self.defmap = None # Base of the defect map
        
    def find_peaks(self, data):
        # Copy the input data
        if data.ndim == 1:
            A = np.array([data])
        else:
            A = np.copy(data)
        Env = np.abs(scsi.hilbert(A, axis = 0))
        self.z_peak = np.argmax(Env, axis = 0) # unitless, index
        
    def get_defect_map(self):
        return self.defmap


class HyperbolaFitTLS2D(HyperbolaFitTLS):
    """ Estimate a defect position by fitting the hyperbola to the measurement data. 
    In this class, the coefficients of the hyperbola polynomial is to be determined
    from the available scan position, x, and the peak position of the measurement
    data, z_peak. 
    
    z = [z_peak_1**2, z_peak_2**2, z_peak_3**2, etc]
    X = coefficient matrix containing 1, xi and xi**2 as its row
      = [[1, x1, x1**2], [1, x2, x2**2], [1, x3, x3**2], etc...]
    w = the LS/TLS solution to find
      = [w0, w1, w2]
      where
          w2 = (z_def/curveture)**2
          w1 = -2* x_def* u2
          w0 = u2* (u1**2/ (4*u2**2) + curveture**2)
      -> this leads to 
          x_def = - w1/ (2* w2)
          z_def = np.sqrt(w0 - w1**2/ (4* w2))
          curvature = np.sqrt(u0 - u1**2/(4*u2**2))
          
    !!!!!! NEW: 200204 !!!!!!
    Hyperbola eq. 
        -(x - x_def)**2 / a**2 + (y - )
    """
    def __init__(self, dz, Nt_offset = 0):
        # Parameter setting
        self.dz = float(dz) #[m]
        self.Nt_offset = Nt_offset
        
        # Base
        self.z_peak = None
        self.x_def = None
        self.z_def = None
        self.Nt_offset = Nt_offset
            

    def solve_TLS(self, x_track):
        """      
        z \approx X* w --- correction ---> z + dz = (X + dX)* w
            => [X+dX z+dz]* [w^T -1]^T = zeros(K) w/ K = # of measurement data
            => [w^T -1]^T lies in the nullspace of [X+dX z+dz]

        [Xz] = U* S* Vh = [U1 u2]* diag(S1, s2)* [V1 v2]^T w/ full rank = 4
        -> Correction of the matrix [Xz] according to the TLS
        [X+dX z+dz] = U1* S1* V1h w/ the rank = 3 
            => v2 is in the nullspace of [X+dX z+dz]
            => [w^T -1]^T = alpha* V2h => w = -1/Vh[-1, -1]* Vh[-1, :-1] 
            
        !!! NEW: 20.02.05 !!!
        Instead of fitting our data to a hyperbola, which actually requires 4 parameters, now we approximate the 
        hyperbola as parabola, i.e. quadratic function as 
            zi = alpha (xi - x_def)**2 + z_def
               = w2* xi**2 + w1* xi + w0
        """
        X = np.zeros((len(x_track), 3))
        X[:, 0] = 1
        for idx in range(len(x_track)):
            X[idx, 1] = x_track[idx]
            X[idx, 2] = x_track[idx]**2
        # Setting for z
        z = (self.z_peak + self.Nt_offset)* self.dz#**2
        # Concatenate [Xz]
        Xz = np.concatenate((X, np.array([z]).T), axis = 1)
        # SVD of Xz
        U, S, Vh = np.linalg.svd(Xz, full_matrices = True)
        # Determine the w
        w = -1/Vh[-1, -1]* Vh[-1, :-1]
        # Calculate xdef, zdef
        self.x_def = - w[1]/ (2* w[2]) #[m]
        self.z_def = w[0] - w[1]**2/(4* w[2]) #[m]

