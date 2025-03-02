# -*- coding: utf-8 -*-
import numpy as np
import json
import matplotlib.pyplot as plt
import time

from tools.datetime_formatter import DateTimeFormatter
from ultrasonic_imaging_python.definitions import units
ureg = units.ureg

from ultrasonic_imaging_python.forward_models.data_synthesizers_progressive import DataGeneratorProgressive2D
from ultrasonic_imaging_python.sako_tools.image_quality_analyzer import ImageQualityAnalyzerMSE

"""
#=================================#
   Iterative Dictionary Update
#=================================#
   
"""
#======================================================================================= Parameter Setting ===#
### Specimen params ###
Nxdata = 40
Ntdata = 880 
t0 = 4.75* 10**-6* ureg.second
#Ntdata = 500 -> t0 = 0* ureg.second

c0 = 6300* ureg.metre/ ureg.second #6300
fS = 80* ureg.megahertz #80
openingangle = 10

dxdata = 0.5*10**-3* ureg.metre
dzdata = 0.5* c0/(fS.to_base_units())
pos_defect = np.array([[20*dxdata.magnitude, 571*dzdata.magnitude]])* ureg.metre 
print(pos_defect)

specimen_params = {
        'Ntdata' : Ntdata,
        'Nxdata' : Nxdata,
        'c0' : c0,
        'fS' : fS,
        'dxdata' : dxdata,
        'dzdata' : dzdata,
        'openingangle' : openingangle,
        'pos_defect' : pos_defect,
        't0' : t0,
        }


### Pulse params ###
fCarrier = 5* ureg.megahertz 
alpha = (2.5* ureg.megahertz)**2 # prameter value from Jan

# Pulse setting for direct Gabor pulse calculation without refmatrix
pulse_params = {
        'fCarrier' : fCarrier,
        'fS' : fS,
        'alpha' : alpha
        }


# Stepsize for the SAFT dictionary 
stepsize = 0.1* ureg.millimetre  #with unit
# For error setting
wavelength = c0/fCarrier.to_base_units() # with unit

# True scan position
p_true = np.array([9])* ureg.millimetre 

#================================================================================================ Function ===#
def minimize_error_with_SGD(atrue, p_track, fwm_prog, analyzer, Niteration, err_max):
    """ Minimize the positional error using stochastic gradient descent(SGD) iteratively. SGD sloves the 
    least-squares problem to a linear matrix eqaution b = A \cdot x. In our case,
        b : (atrue - amodel)
            difference b/w the measured A-Scan(atrue) and the modeled A-Scan based on the p_est(amodel)
        A : curr_grad
            derivative of a modeled A-Scan
        x : positional error [m], unitless
    Since the derivative of a modeled A-Scan is not linear, SGD works properly, only when the error is within
    the certain range which varies with the scan position (ptrue) and other measurement setups.
    
    Parameters
    ----------
        atrue : array-like (Nt)
            Measured A-Scan
        p_track : float, [m] unitless!
            Tracked scan position (p_track != p_true)
        fwm_prog : class
           DataGeneratorProgressive2D class, the first four steps should be done beforehand 
           (Cf. DataGeneratorProgressive2D)
        analyzer : class
            ImageQualityAnalyzerMSE class, initialization should be done with atrue beforehand
        Niteration : int
            Max number of iteration for SGD
    
    Returns
    -------
        curr_p_est : float, [m] unitless!
            Optimized scan position through iterative SGD
        aopt : array (Nt)
            1st Taylor approximation with the optimized scan position and the corresponding error
        se : float
            Normlized square error (Cf. ImageQualityAnalyzerMSE)
        
    """    
    curr_p_est = float(p_track) #[m], unitless
    # The first column of A, b -> all 0, as it corresponds to err_x = 0
    A = np.zeros((atrue.shape[0], 1))
    b = np.zeros((atrue.shape[0], 1))
    
    # Set the initial se_min
    se_min = 1.0

    for n in range(Niteration):
        print('*** Iteration No.{} ***'.format(n))
        fwm_prog.set_reflectivity_matrix(np.array([curr_p_est])*ureg.metre, oracle_cheating = True)
        Hmodel = fwm_prog.get_impulse_response_dictionary(without_refmatrix = True)
        amodel = fwm_prog.get_single_ascan(Hmodel)
        del Hmodel
        Hderiv = fwm_prog.get_derivative_dict(deriv_wrt_tau = False)
        curr_grad = fwm_prog.get_single_ascan(Hderiv)
        del Hderiv
        
        # Update A and b
        A = np.append(A, np.array([curr_grad]).transpose(), axis = 1)
        b = np.append(b, np.array([atrue - amodel]).transpose(), axis = 1) 
        # Solve b = A \cdot x, only the first one is returned, as others (e.g. residual) are irrelevant
        err_xest = np.linalg.lstsq(A, b, rcond=None)[0] # rcond=None is given, becuase of the warning
        print('Estimated error : {}'.format(err_xest[-1, n+1]))

        curr_aopt = amodel + curr_grad * err_xest[-1, n+1]
        
        if abs(err_xest[-1, n+1]) > err_max:
            curr_se = 1.0
        else:
            curr_se = analyzer.get_mse(np.array([curr_aopt]))
            curr_p_est = curr_p_est - err_xest[-1, n+1]
        
        print('Optimized scan position : {}mm'.format(round(curr_p_est* 10**3, 3)))
        print('Current SE : {}'.format(curr_se))
        
        if se_min >= curr_se:
            se_min = curr_se
            aopt = curr_aopt
            p_est = curr_p_est
        
        if se_min <= 0.01:
            break
        
        elif curr_p_est < 0:
            break
    
    print("Final error:", se_min)
    return p_est, aopt, se_min



#================================================================================================== a_true ===#
fwm_prog = DataGeneratorProgressive2D(specimen_params, stepsize)
fwm_prog.set_pulse_parameter(pulse_params)
fwm_prog.unit_handling_specimen_params()
fwm_prog.vectorize_defect_map()

fwm_prog.set_reflectivity_matrix(p_true, oracle_cheating = True)
adelta = fwm_prog.get_single_ascan(fwm_prog.refmatrix)
Htrue = fwm_prog.get_impulse_response_dictionary(without_refmatrix = True)
atrue = fwm_prog.get_single_ascan(Htrue)
# Get ToF for a_opt2 calculation (-> the difference of the ToF is required )
tof_true = fwm_prog.tof_calculator.tof #[S], unitless
# Delete teh dictionary to reduce the memory consumption
del Htrue

# Call ImageQualityAnalyzer for SE calculation
analyzer = ImageQualityAnalyzerMSE(np.array([atrue]))

#===================================================================================================== SGD ===#
# Set p_track
error = 0.8* wavelength
p_track = ((p_true + error).to_base_units()).magnitude #[m], unitless
print(error)

# Choose Niteration
Niteration = 20

p_est, aopt, se = minimize_error_with_SGD(atrue, p_track, fwm_prog, analyzer, Niteration, 2*wavelength.magnitude)

plt.plot(atrue)
plt.plot(aopt)
plt.legend(['atrue', 'aopt'])


