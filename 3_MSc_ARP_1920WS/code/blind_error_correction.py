# -*- coding: utf-8 -*-
"""
"""
import numpy as np

import defect_map_handling
import tof_calculator
import dictionary_former
import hyperbola_fit_TLS
import time
from display_time import display_time
from gaussian_blur import gaussian_blur 


class BlindErrorCorrection2D():
    
    def __init__(self, fwm_params, Nt_offset = 0, r = 0):
        r""" Constructor
        Parameters
        ----------
            fwm_params: dict
                Containing FWM parameters shown below              
        """
        # FWM parameter setting
        self.Nx = fwm_params['Nx'] # limited due to the opening angle
        self.Nz = fwm_params['Nz'] 
        self.Nt = self.Nz
        self.c0 = fwm_params['c0']  #[m/S]
        self.fS = fwm_params['fS']  #[Hz] 
        self.fC = fwm_params['fC']  #[Hz] 
        self.alpha = fwm_params['alpha']  #[Hz]**2
        self.opening_angle = fwm_params['opening_angle']
        self.dx = fwm_params['dx']  #[m]
        self.dz = 0.5* self.c0/(self.fS) #[m]
        self.Nt_offset = Nt_offset
        self.r = r # unitless, -1...1, Chirp rate
        self.M = self.Nt - self.Nt_offset # Vertical dimension adjusted to the ROI
        
        # Base for dictionary fomation class
        self.dformer = dictionary_former.DictionaryFormer(self.Nt, self.fS, self.fC, self.alpha, self.Nt_offset, 
                                                          self.r)
        # Base for the hyperbola fitting
        self.hypfit = None
        self.x_def = None
        self.z_def = None
        # Base for the global class parameters
        self.a_true = None
        self.defmap_true = None
        self.b_hat = None # estimated defect positions as a vector
        self.p_true = None #[m]
        self.a_err = None
        self.grada_model = None
        self.err_max = None
        
        
    def defectmap_generation(self, p_def_idx):
        r""" Constructor
        Parameters
        ----------
            p_def_idx: int/float
                Indicating where a defect is located
                (Currently this class supports only single defect case)              
        """
        # Defect map generation
        p_def = np.array([p_def_idx[0]*self.dx, p_def_idx[1]*self.dz])
        dmh = defect_map_handling.DefectMapSingleDefect2D(p_def, self.Nx, self.Nz, self.dx, self.dz)
        dmh.generate_defect_map(self.Nt_offset)
        self.defmap_true = dmh.get_defect_map()
        
    
    def dictionary_formation(self, p, with_Jacobian = False, with_envelope = False, Nx_dict = None):
        r""" Calculate SAFT matirix. If desired, its Jacobian is also returned.
        Parameters
        ----------
            p: np.ndarray, (arbitrary, 2)
                Positions to calculate the dictionary
                p.shape[0] can be arbitrary chosen, whereas p.shape[1] shuld be 2(-> x, z)
                    e.g. p = [[x1, z1], [x1, z2], [x1, z3].....]
            Nx_dict: int  (None by default)
                In case the dictionar size to besmaller than the measurement ROI (e.g. ROI contains zero-vectors),
                adjust the dictionary size
        """
        # Dimension adjustment
        if Nx_dict is None:
            Nx_dict = np.copy(self.Nx)
        # ToF calculation
        tofcalc = tof_calculator.ToFforDictionary2D(self.c0, Nx_dict, self.Nz, self.dx, self.dz, p, self.Nt_offset,
                                                    self.opening_angle)
        tofcalc.calculate_tof(calc_grad = with_Jacobian)
        tof = tofcalc.get_tof()
        apf = tofcalc.get_apodization_factor()

        if with_Jacobian == False:
            grad_tof = None
        else:
            grad_tof = tofcalc.get_grad_tof()
        del tofcalc
        
        # Dictionary formation 
        self.dformer.generate_dictionary(tof, grad_tof, with_envelope = with_envelope)
        H = self.dformer.get_SAFT_matrix(apf)
        
        if with_Jacobian == False:
            del tof, self.dformer.H, self.dformer.G
            return H
        else:
            J = self.dformer.get_Jacobian_matrix()
            del tof, grad_tof, self.dformer.H, self.dformer.J, self.dformer.G
            return H, J
        
    def partially_scanned_dictionary(self, p_scan_idx):
        """ Calculate the measurment dictionary for the data generation, where the A-Scans are taken only at the 
        selected measurement grid points.
        
        Parameters
        ----------
            p_scan_idx: np.ndarray(int)
                Array of the indices where the A-Scans should be generated
                
        Returns
        -------
            H: np.ndarray w/ size = L x L with L = self.M* self.Nx
                Measuremnt dictionary of the full size (but only partially computed according to the p_scan_idx)
        """
        L = self.M* self.Nx # Dimension of the dictioary for each position
        H = np.zeros((L, L))
        for x in range(self.Nx):
            if x in p_scan_idx:
                p = np.array([[x* self.dx, 0]])
                start = x* self.M
                end = (x+1)*self.M
                H[start:end, :] = self.dictionary_formation(p)
        return H
    
    def data_generation(self, p_scan_idx = None):
        r""" Calculate a set of synthetic data
        
        Parameters
        ----------
            p_scan_idx: np.array (int) (None by default)
                If given, A-Scans are calculated only for the selected grid points
        
        """
        self.p_true = np.zeros((self.Nx, 2))
        self.p_true[:, 0] = np.arange(self.Nx)* self.dx
        if p_scan_idx is None:
            H = self.dictionary_formation(self.p_true)
        else:
            H = self.partially_scanned_dictionary(p_scan_idx)
        self.a_true = np.dot(H, self.defmap_true)
                    
    
    def get_data(self, ret_Atrue = False):
        if ret_Atrue == False:
            return self.a_true
        else:
            A_true = np.reshape(self.a_true, (self.M, self.p_true.shape[0]), 'F')
            return self.a_true, A_true
    
#=============================================================================================== Hyperbola Fit TLS ====#    
    def estimate_defect_positions(self, A_roi, p):
        r""" Estimate the defect position via TLS hyperbola fit. 
        Parameters
        ----------
            p: np.ndarray
                Positions to calculate the dictionary (only x element)
                e.g. p = [x1, x2, x3, .....]
        """
        self.hypfit = hyperbola_fit_TLS.HyperbolaFitTLS2D(self.dz, self.Nt_offset)
        self.hypfit.find_peaks(A_roi)
        self.hypfit.solve_TLS(p)
        self.x_def = self.hypfit.x_def #[m]
        self.z_def = self.hypfit.z_def #[m]
        del self.hypfit
        
        
    def defect_map_conversion(self, Nx_raw = None, col_eliminate = None, blur = False, sigma_z = None, sigma_x = None):
        r"""
        Parameters:
        -----------
            Nx_raw: int 
                Dimension of the raw data (including zero columns) to avoid that the estimated defect position would
                look like it is beyond the ROI
                    e.g.: Nx_raw = 100 and self.Nx = Nx_roi = 40
                          x_def = 55* dx, z_def = somewhere
                          -> generating the defect map based on self.Nx makes it look like that the defect is 
                          outside the ROI (because x_def_idx = 55 > Nx_roi)
            col_eliminate: np.ndarray, list
                Columns to eliminate (because of the zero-vectors)
                
        """
        if Nx_raw is None:
            Nx_dm = np.copy(self.Nx)
        else:
            Nx_dm = np.copy(Nx_raw)
        # Point source
        if blur == False:
            if (self.x_def >= Nx_dm* self.dx or self.z_def >= self.Nz* self.dz):
                raise ValueError('BlindErrorCorrection2D: estimated defect position is outside of the ROI!')
            else:
                # Defect map based on the size of the raw data (with Nx_raw)
                dmh = defect_map_handling.DefectMapSingleDefect2D([self.x_def, self.z_def], Nx_dm, self.Nz, self.dx, 
                                                                  self.dz)
                dmh.generate_defect_map_multidim(self.Nt_offset, col_eliminate) # the defmap is adjusted to the ROI!!!
                self.b_hat = dmh.get_defect_map_1D() 
        # Physical-sized scatterer
        else:
            x_def_idx = self.x_def / self.dx
            z_def_idx = self.z_def / self.dz - self.Nt_offset
            # Estimated defect map as a matriy (2D)
            B_hat = gaussian_blur(self.M, Nx_dm, [z_def_idx, x_def_idx], sigma_z, sigma_x)
            # Adjust to teh ROI: eliminate the zero-columns
            if col_eliminate is not None:
                B_hat = np.delete(B_hat, col_eliminate, axis = 1)
            # Unfold the defect map (2D -> 1D)
            self.b_hat = B_hat.flatten('F')
            # Remove very small non-zero entry
            idx_toosmall = np.argwhere(self.b_hat <= 10**-3)
            self.b_hat[idx_toosmall] = 0.0
        

        
#============================================================================ Iterative Error Correction w/ Newton ====#    
    def error_correction(self, p, a_true, Niteration, epsilon, err_max, triangular = False):
        r""" 
        Parameters
        ----------
            p: np.ndarray(Nreal)
                Mutiple erronous tracked positions for a single grid 
            a_true: np.ndarray(M* Nreal)
                Measurement data of a single grid assigned to the tracked positions, p
            Niteration: int
                # of iterations for Newton method 
            epsilon: float
                Target value for Newton method, which is used as the break condition. 
            err_max: float, positive
                (Upper) bound for the tracking error (which is uniformply distributed)
                i.e. -err_max <= delta_x <= err_max
        """
        self.err_max = float(err_max)
        # Initial setting
        p_hat = np.copy(p)
        x_hat = np.copy(p_hat[:, 0])
        if triangular == False:
            deltax_hat = np.random.uniform(-self.err_max, self.err_max, x_hat.shape[0])
        else:
            deltax_hat = np.random.triangular(-self.err_max, 0, self.err_max, x_hat.shape[0])
        # Base to store the min value setting
        fx_min = -1
        x_hat_min = np.zeros(x_hat.shape[0])
        deltax_hat_min = np.zeros(x_hat.shape[0])
        a_opt = np.zeros(a_true.shape[0])
        
        # Newton method -> deltax_hat          
        for n in range(Niteration):    
            print('Iteration No.{}'.format(n))
            H, J = self.dictionary_formation(p_hat, with_Jacobian = True, Nx_dict = p.shape[0])
            # Modeled A-Scan
            a_model = np.dot(H, self.b_hat)
            del H
            # Gradient of the odeled A-Scan
            self.grada_model = np.dot(J, self.b_hat)
            del J
            # Error b/w the measurement data (a_true) and the modeled A-Scan
            self.a_err = a_true - a_model
            
            # Newton method
            fx_preNewton = self.fx(deltax_hat)
            deltax_hat = self.Newton_method(deltax_hat)
            fx_postNewton = self.fx(deltax_hat)
            print('Pre-Newton fx(deltax_hat): {}'.format(round(fx_preNewton, 5)))
            print('Post-Newton fx(deltax_hat): {}'.format(round(fx_postNewton, 5)))
            print('Max absolute error: {}mm'.format(round(abs(deltax_hat).max()* 10**3, 5)))
            print('1st element: {}mm'.format(round(x_hat[0]*10**3, 3)))
            print('------------------')
            
            if fx_min < 0 or fx_min > fx_postNewton:
                fx_min = fx_postNewton
                x_hat_min = np.copy(x_hat)
                deltax_hat_min = np.copy(deltax_hat)
                #a_opt = a_model + np.dot(np.kron(np.diag(deltax_hat), np.identity(self.M)), self.grada_model)
                a_opt = a_model + np.repeat(deltax_hat, self.M)* self.grada_model # faster than Kronecker 
                
            # Break conditions: either reached the target or converged
            if fx_postNewton <= epsilon or (fx_preNewton - fx_postNewton) <= 10**-6:
                break
            # Update for the next iteration
            else:
                x_hat = x_hat - deltax_hat
                p_hat[:, 0] = np.copy(x_hat) 
            
        return x_hat_min, deltax_hat_min, a_opt
    
    
    def check_constraints(self, x_postNewton):
        r""" Check whether the obtained solution is a feasible solution or not.
        Here, our constraint is
            delta_x <= self.err_max
        """
        for idx, element in enumerate(x_postNewton):
            if np.abs(element) >= self.err_max: # -> which factor should we choose?
                print('!! No. {} is too large !!'.format(idx))
                print('Too large element: {}mm'.format(round(element*10**3, 5)))
                x_postNewton[idx] = self.err_max* (element/abs(element)) # to keep the same sign
                print('After correction: {}mm'.format(round(x_postNewton[idx]*10**3, 5)))
        return x_postNewton
        
    
    def Newton_method(self, x):
        r""" Gradient of the cost function
        Prameters
        ---------
            x: np.ndarray
                -> in this context: x = deltax_hat = estimate of the tracking error
        """
        grad = self.grad_fx(x)
        inv_hess = self.inv_hess_fx(x)
        x_postNewton = x - np.dot(inv_hess, grad)
        # Constraints
        solution = self.check_constraints(x_postNewton)
        return solution
    
    
    def fx(self, x):
        r""" Cost function for the Newton method
        Prameters
        ---------
            x: np.ndarray
                -> in this context: x = deltax_hat = estimate of the tracking error
        """
        
        jacobi = np.dot(np.kron(np.diag(x), np.identity(self.M)), self.grada_model)
        #jacobi = np.repeat(x, self.M)* self.grada_model # faster than Kronecker 
        return (np.linalg.norm(self.a_err + jacobi))**2

    
    def grad_fx(self, x):
        r""" Gradient of the cost function
        Prameters
        ---------
            x: np.ndarray
                -> in this context: x = deltax_hat = estimate of the tracking error
        """
        grad_fx = np.zeros(x.shape[0])
        for k in range(x.shape[0]):
            start = k* self.M
            end = (k+1)* self.M
            grad_fx[k] = 2*np.sum((self.a_err[start:end] + self.grada_model[start:end]* x[k])* 
                         self.grada_model[start:end])
        return grad_fx
    
    def inv_hess_fx(self, x):
        r""" Inverse Hessian of the cost function
        Prameters
        ---------
            x: np.ndarray(Nreal)
                -> deltax_hat = estimate of the tracking error
            grada_model: np.ndarray(M* Nreal)
                Gradient of the modeled A-Scans, a_true, w.r.t. the position
        """
        diag_elements = np.zeros(x.shape[0])
        for k in range(x.shape[0]):
            start = k* self.M
            end = (k+1)* self.M
            element = 2*np.sum(self.grada_model[start:end]**2)
            if element == 0:
                diag_elements[k] = 0
            else:
                diag_elements[k] = 1/element
        inv_hess_fx = np.diag(diag_elements)
        return inv_hess_fx   
    

#======================================================================================= Data & Position Selection ====#
    def position_selection(self, se, x_hat, deltax_hat):
        r""" Select one estimated scan position from the corrected tracking position, x_hat. The position with the 
        smallest model error (i.e. variance to the true A-Scan) and the estimated tracking error.
        If there are 2+ candidates, choose the first one with the minimal estimated tracking error.
        
        Parameters
        ----------
            se: np.array(Nreal)
                Model error b/w the true measurement data (for a particular grid point) and the modeled A-Scans
                (here modeled A-Scan = np.dot(H(x_hat_i, deltax_hat_i), b_hat))
            x_hat: np.ndarray(Nreal)
                "Corrected" tracking positions
            deltax_hat: np.ndarray(Nreal)
                Estimated tracking error after the error correction
                
        Returns
        -------
            x_est: float
                Estimated scan position for the given measurment data (of a particular grid point)
            deltax_est: float
                Estimated tracking error which corresponds to the selected scan position, x_est
        """
        semin_index = np.where(se == se.min())[0]
        print(semin_index)
        if len(semin_index) == 1:
            x_est = x_hat[semin_index]
            deltax_est = deltax_hat[semin_index]
        else:
            deltaxmin_idx = semin_index[np.where(deltax_hat[semin_index] == deltax_hat[semin_index].min())[0]]
            x_est = x_hat[deltaxmin_idx[0]]
            deltax_est = deltax_hat[deltaxmin_idx[0]]
        
        return x_est, deltax_est
        
        