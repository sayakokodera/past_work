import numpy as np
import pulse_former 

class DictionaryFormer():
    def __init__(self, Nt, fS, fC, alpha, Nt_offset = 0, r = 0):
        self.pformer = pulse_former.PulseFormerGabor(Nt, fS, fC, alpha, r)
        self.Nt_offset = Nt_offset # time offset(= dt* Nt_offset) to remove 
        self.Nt = int(Nt)
        self.with_envelope = None
        # Dictionary dimension
        self.M = self.Nt - self.Nt_offset
        self.K = None 
        self.L = None
        # Base
        self.H = None # Base of the SAFT matrix
        self.J = None # Base of the Jacobian matrix
        self.G = None # Base of the apodization matrix
 
                   
    def generate_dictionary(self, tof, grad_tof = None, with_envelope = False):
        r"""
        Parameters
        ----------
            tof: np.ndarray [S, but no ureg] w/ size = K x L
                ToF for the given scan positions 
                K = # of scan positions -> rows correspond to the scan positions
                L = # of grid points in ROI -> columns correscond to the possible defect positions
            grad_tof: np.ndarray [S/m] w/ size = K x L
                Gradient of ToF w.r.t. the scan positions
        """
        self.with_envelope = with_envelope
        self.K = tof.shape[0]
        self.L = tof.shape[1]
        # SAFT matrix
        self.H = np.apply_along_axis(self.calculate_column_vec_H, 0, np.array([tof.flatten('F')]))
        self.H = np.reshape(self.H, (self.M* self.K, self.L), 'F')
        # Jacobian matrix
        if grad_tof is not None:
            self.J = np.apply_along_axis(self.calculate_column_vec_J, 0, np.array([grad_tof.flatten('F')]))
            self.J = np.reshape(self.J, (self.M* self.K, self.L), 'F')
 
                  
    def calculate_column_vec_H(self, tau):
        if self.with_envelope == True:
            self.pformer.calculate_envelope(tau)
            return self.pformer.get_envelop(self.Nt_offset)
        else:
            self.pformer.calculate_ip_pulse(tau)
            return self.pformer.get_ip_pulse(self.Nt_offset)
    
    def calculate_column_vec_J(self, grad_tau):
        if self.with_envelope == True:
            self.pformer.calculate_grad_envelope(grad_tau)
            return self.pformer.get_grad_envelop(self.Nt_offset)
        else:
            self.pformer.calculate_grad_ip(grad_tau)
            return self.pformer.get_grad_ip(self.Nt_offset)
        

    def get_apodization_matrix(self, apf):
        r"""
        Parameter:
        ----------
            apf: np.ndarray w/ the size = K x L
                Array containing the apodization factor for each scan-defect combination
        Obtain:
        -------
            G: np.ndarray w/ the size = L*M x M*K
                Apodization matrix which can be directly multiplied with the SAFT or Jacobian matrix
                *** np.repeat is much faster and less expensive than the dot product with the Kronecker product!
        """
        return np.repeat(apf, self.M, axis = 0)
        
   
    def get_SAFT_matrix(self, apf = None):
        if apf is None:
            self.G = None
            return self.H
        else:
            self.G = self.get_apodization_matrix(apf)
            return self.G* self.H
    
    
    def get_Jacobian_matrix(self):
        if self.G is None:
            return self.J
        else:
            return self.G* self.J
        
 
if __name__ == '__main__':
    import tof_calculator
    import time
    
    # Parameter setting
    Nx = 20
    Nz = 1300
    Nt_offset = 1200
    Nz_roi = Nz - Nt_offset
    c0 = 6300 #[m/s]
    fS = 80*10**6 #[Hz]
    fC = 5*10**6 #[Hz]
    alpha = 20*10**12 #[Hz]**2
    opening_angle = 20
    #t_offset = Nt_offset/fS #[s]
    dx = 0.5*10**-3 #[m]
    dz = 0.5* c0/(fS)
    p_scan = np.zeros((Nx, 2))
    p_scan[:, 0] = np.arange(Nx)* dx
   
    # ToF with Nt_offset
    tofcalc = tof_calculator.ToFforDictionary2D(c0, Nx, Nz, dx, dz, p_scan, Nt_offset, opening_angle)
    tofcalc.calculate_tof(calc_grad = False)
    tof1 = tofcalc.get_tof()
    apf = tofcalc.get_apodization_factor()
    
    # Dictionary fast
    start2 = time.time()
    dformer2 = DictionaryFormer(Nz, fS, fC, alpha, Nt_offset)
    dformer2.generate_dictionary(tof1, with_envelope = True)
    H2 = dformer2.get_SAFT_matrix()
    print('H2 = {}S'.format(round(time.time() - start2, 3)))
    
    # w/ Apodization
    H2a = dformer2.get_SAFT_matrix(apf)   
    
