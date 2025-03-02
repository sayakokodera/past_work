import numpy as np
import time

class ToFCalculator():
    """ ToF calculator for a single defect and single scan position. 
    """
    def __init__(self):
        self.c0 = None
        self.tof = None
        self.grad_tof = None
        
    def get_angle(self, p_def, p_scan):
        origin = np.array(p_scan)
        vec1 = np.array([0.0, 1.0])
        vec2 = p_def - origin
        # angle b/w two vectors
        num = np.dot(vec1, vec2)
        den = np.linalg.norm(vec2)
        if den == 0:
            angle = 0
        else:
            angle = np.rad2deg(np.arccos(num/den))
        return angle
        
    
    def get_tau(self, p_def, p_scan): # Universal, possible to use in 3D cases!
        return 2*np.linalg.norm(p_def - p_scan)/ self.c0 #[S], unitless
    
    
    def get_tof(self):
        return self.tof
    
    
    def get_grad_tof(self):
        return self.grad_tof

    
class ToFforDictionary2D(ToFCalculator):
    """ ToF calculator for a dictionary formation. 
        Size of tof = K x L
            K = # of scan positions -> rows correspond to the scan positions
            L = # of grid points in ROI -> columns correscond to the possible defect positions
    """
    def __init__(self, c0, Nx, Nz, dx, dz, p_scan_all, Nt_offset = 0, opening_angle = 180):
        self.c0 = float(c0)
        self.Nx = int(Nx)
        self.Nz = int(Nz)
        self.dx = float(dx)
        self.dz = float(dz)
        # Vertical adjsutment of ROI (= removing t_offset)
        self.Nt_offset = int(Nt_offset)
        
        # Check the dimension of p_scan_all 
        if p_scan_all.ndim == 1: # Without handling, x and z are handled separately
            self.p_scan_all = np.array([p_scan_all]) # in [m], unitless
        else:
            self.p_scan_all = np.array(p_scan_all) # in [m], unitless
        # ToF size according to the ROI -> tof size = K x L
        self.M = self.Nz - self.Nt_offset
        self.K = self.p_scan_all.shape[0]
        self.L = self.M* self.Nx
        # Horizontal adjustment of ROI (= only the region within the opening angle)
        self.opening_angle = opening_angle
        # undefined parameters
        self.calc_grad = None
        self.p_all = None

    
    def get_grad_tau(self, x_def, x_scan, tau):
        if tau == 0.0:
            return 0.0
        else:
            return -4* (x_def - x_scan)/ (tau* self.c0**2) # multiply by "-"??? -> check!! (19.12.21)
                        
                        
    def calculate_tof(self, calc_grad = False):
        """
        Parameter
        ---------
        calc_grad: boolean
            True, if gradient (derivative) of tau should be calculated
        opening_angle: int, float in [deg]
            To restrict the region to consider, default value is 180 [deg]
            
        Test
        ----
        0.5* tau* c0 = np.linalg.norm(p_def - p_scan)
        """
        self.calc_grad = calc_grad
        # Postion setting
        self.position_vector()
        # Without grad_tof
        if self.calc_grad == False:
            self.tof = np.apply_along_axis(self.calculate_tau, 1, self.p_all)
            self.tof = np.reshape(self.tof, (self.K, self.L), 'F')
        # With grad_tof
        else:
            output = np.apply_along_axis(self.calculate_tau, 1, self.p_all) # apply_along_axis returns only one array!!
            self.tof, self.grad_tof = output[:, 0], output[:, 1]
            self.tof = np.reshape(self.tof, (self.K, self.L), 'F')
            self.grad_tof = np.reshape(self.grad_tof, (self.K, self.L), 'F')
            del output

    
    def position_vector(self):
        r""" Return a vector containing all possible position combinations for tof_calculation using apply_along_axis.
        p_all is with the size = K*L x 2
        (e.g.)  K = 2, Nx = 2, M = 3 -> L = M* Nx = 6
            p_all = [
                    [xscan_0, zscan_0, xdef_0, zdef_0],
                    [xscan_1, zscan_1, xdef_0, zdef_0],
                    [xscan_0, zscan_0, xdef_0, zdef_1],
                    [xscan_1, zscan_1, xdef_0, zdef_1],
                    [xscan_0, zscan_0, xdef_0, zdef_2],
                    [xscan_1, zscan_1, xdef_0, zdef_2],
                    [xscan_0, zscan_0, xdef_1, zdef_0],
                    [xscan_1, zscan_1, xdef_1, zdef_0],
                    [xscan_0, zscan_0, xdef_1, zdef_1],
                    [xscan_1, zscan_1, xdef_1, zdef_1],
                    [xscan_0, zscan_0, xdef_1, zdef_2],
                    [xscan_1, zscan_1, xdef_1, zdef_2],
                    ]
        """
        self.p_all = np.zeros((self.K* self.L, 4)) # = [x_scan, z_scan, x_def, z_def]
        # Scan positions
        self.p_all[:, :2] = np.reshape(np.stack([self.p_scan_all for _ in range(self.L)], axis = 0), (self.K* self.L, 2))
        # Defect positions
        x_def_base = np.repeat(np.arange(self.Nx), self.M)* self.dx
        z_def_base = (np.stack([np.arange(self.M) for _ in range(self.Nx)], axis = 0).flatten('C') \
                        + self.Nt_offset)* self.dz
        self.p_all[:, 2:] = np.repeat(np.array([x_def_base, z_def_base]).T, self.K, axis = 0)

    
    def calculate_tau(self, p):
        p_scan = p[:2]
        p_def = p[2:]
        # Constraint: opening angle
        # -> Consider the region only within the opening angle
        angle = self.get_angle(p_def, p_scan)
        if angle <= 0.5* self.opening_angle:
            tau = self.get_tau(p_def, p_scan)
        else:
            tau = 0.0 # float! otherwise self.tof becomes an array of int!
        
        # Without grad_tau
        if self.calc_grad == False:
            return tau
        # With grad_tau
        else:
            if tau == 0.0:
                grad_tau = 0.0
            else:
                grad_tau = self.get_grad_tau(p_def[0], p_scan[0], tau)
            return tau, grad_tau
        
                        
    def apodization_function(self, p):
        r""" Function to calculate the apodization factor to model the attenuation in the physical signal response.
        Apodization is determined by the scan position (x_scan, z_scan), the defect position (x_deef, z_def)
        and the transducer openine angle.
            g(x_scan, z_scan, x_def, z_def, opening_angle) = 
                np.exp(-(x_scan - x_def)**2 / (tan(opening_angle)* (z_scan - z_def))**2)
        """
        x_scan, z_scan, x_def, z_def = p
        return np.exp(-(x_scan - x_def)**2 / (np.tan(np.deg2rad(self.opening_angle))* (z_scan - z_def))**2)
    
    
    def get_apodization_factor(self):
        r""" Return the apodization factors for all possible scan-defect combination for the dictionary formation.
        Return
        ------
            apf_arr: np.ndarray with size = K x L
        """
        apf_vec = np.apply_along_axis(self.apodization_function, 1, self.p_all)
        return np.reshape(apf_vec, (self.K, self.L), 'F')
        



if __name__ == '__main__':
    # Parameter setting
    Nx = 300
    Nz = 1300
    c0 = 6300 #[m/s]
    fS = 80*10**6 #[Hz]
    #t_offset = Nt_offset/fS #[s]
    dxdata = 0.5*10**-3 #[m]
    dzdata = 0.5* c0/(fS)
    p_scan = np.array([[2*dxdata, 0*dzdata], [17*dxdata, 0*dzdata]])
    # ROI adjustment
    Nt_offset = 1100
    Nz_roi = Nz - Nt_offset
    
    # Tof calculation
    start2 = time.time()
    tofcalc2 = ToFforDictionary2D(c0, Nx, Nz, dxdata, dzdata, p_scan, Nt_offset = Nt_offset, opening_angle = 30)
    tofcalc2.calculate_tof(calc_grad = False)
    tof2 = tofcalc2.get_tof()
    # Apodization
    apf = tofcalc2.get_apodization_factor()

