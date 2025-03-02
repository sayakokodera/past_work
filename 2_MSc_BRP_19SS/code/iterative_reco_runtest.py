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
from ultrasonic_imaging_python.sako_tools.parameter_dictionary_exporter import ParameterDictionaryExporter



#======================================================================================================== Function ===#
def minimize_error_with_SGD(atrue, p_track, fwm_prog, analyzer, Niteration, err_max):
    """ Minimize the positional error using stochastic gradient descent(SGD) iteratively. SGD sloves the 
    least-squares problem to a linear matrix eqaution b = A \cdot x. In our case,
        b : collection of (atrue - amodel)
            difference b/w the measured A-Scan(atrue) and the modeled A-Scan based on the p_est(amodel)
        A : collection of curr_grad
            derivative of modeled A-Scans
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
    print('########')
    print('Initial tracked position: {}mm'.format(curr_p_est* 10**3))
    print('########')
    # The first column of A, b -> all 0, as it corresponds to err_x = 0
    A = np.zeros((atrue.shape[0], 1))
    b = np.zeros((atrue.shape[0], 1))
    
    # Set the initial se_min
    se_min = 1.0

    for n in range(Niteration):
        print('Iteration No.{}'.format(n))
        fwm_prog.set_reflectivity_matrix(np.array([curr_p_est])*ureg.metre, oracle_cheating = True)
        Hmodel = fwm_prog.get_impulse_response_dictionary(without_refmatrix = True)
        amodel = fwm_prog.get_single_ascan(Hmodel)
        del Hmodel
        Hderiv = fwm_prog.get_dictionary_derivative(deriv_wrt_tau = False)
        curr_grad = -fwm_prog.get_single_ascan(Hderiv)
        del Hderiv
        
        # Update A and b
        A = np.append(A, np.array([curr_grad]).transpose(), axis = 1)
        b = np.append(b, np.array([atrue - amodel]).transpose(), axis = 1) 
        # Solve b = A \cdot x, only the first one is returned, as others (e.g. residual) are irrelevant
        err_xest = np.linalg.lstsq(A, b, rcond=None)[0] # rcond=None is given, becuase of the warning
        print('Size of err_est : {}'.format(err_xest[:, -1]))
        print('Estimated error : {}mm'.format(round(err_xest[-1, -1]* 10**3, 3)))

        curr_aopt = amodel + curr_grad * err_xest[-1, -1]
        
        if abs(err_xest[-1, -1]) > err_max:
            curr_se = 1.0
        else:
            curr_se = analyzer.get_mse(np.array([curr_aopt]))
            curr_p_est = curr_p_est - err_xest[-1, -1]
        
        print('Optimized scan position : {}mm'.format(round(curr_p_est* 10**3, 3)))
        print('Current SE : {}'.format(curr_se))
        
        if se_min >= curr_se:
            se_min = float(curr_se)
            aopt = np.array(curr_aopt)
            p_est = float(curr_p_est + err_xest[-1, -1]) # add the error to reverse the error-adjustment
            perr_est = err_xest[-1, -1]
        
        if se_min <= 0.01:
            break
        
        elif curr_p_est < 0:
            break
    
    print("Final error:", round(se_min, 5))
    print('Final scan position: {}mm'.format(round(p_est* 10**3, 3)))
    return p_est, perr_est

#================================================================================================ Parameter Setting ===#
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
alpha = 1.5* ureg.megahertz**2 # prameter value from Jan

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
wvl_magmm = wavelength.to('millimetre').magnitude

# True scan position
p_true_all = np.array([9])#np.array([2.5, 5, 7.5, 9, 9.37])

# Set the error range
perr_norm = 1
perr = perr_norm* wavelength
perr_func = '{}* wavelength'.format(perr_norm)
err_max = 5* wavelength.magnitude

# Choose Niteration for SGD
Niteration = 15


#============================================================================================================== GD ===#
# Initial setting
p_opt_all = np.zeros(p_true_all.shape)
perr_all = np.zeros(p_true_all.shape)

# Get time
start_all = time.time() 

fwm_prog = DataGeneratorProgressive2D(specimen_params, stepsize)
fwm_prog.set_pulse_parameter(pulse_params)
fwm_prog.unit_handling_specimen_params()
fwm_prog.vectorize_defect_map()

### Iteration over p_true ###
for p_idx in range(p_true_all.shape[0]):
    print('*** Position No.{}***'.format(p_idx))
    # Get time
    start = time.time()
    
    p_true = np.array([p_true_all[p_idx]])* ureg.millimetre 
    fwm_prog.set_reflectivity_matrix(p_true, oracle_cheating = True)
    adelta = fwm_prog.get_single_ascan(fwm_prog.refmatrix)
    Htrue = fwm_prog.get_impulse_response_dictionary(without_refmatrix = True)
    atrue = fwm_prog.get_single_ascan(Htrue)
    # Delete teh dictionary to reduce the memory consumption
    del Htrue
    # Get p_true without unit -> for evaulating the error correction
    p_true_unitless = fwm_prog.x_scan
    
    # Call ImageQualityAnalyzer for SE calculation
    analyzer = ImageQualityAnalyzerMSE(np.array([atrue])) 
    
    # Set p_track
    p_track = ((p_true + perr).to_base_units()).magnitude
    # Estimate error
    p_est, perr_est = minimize_error_with_SGD(atrue, p_track, fwm_prog, analyzer, Niteration, err_max)
    # Store the obtained results
    p_opt_all[p_idx] = p_est
    perr_all[p_idx] = perr_est
    
    # Get time
    stop = time.time() - start
    print('******')
    print('Calculation for Position No.{} takes...'.format(p_idx))
    print('{} s'.format(round(stop, 3)))
    print('******')

# Get time
comp_all = time.time() - start_all

print('All calculation takes...')
if comp_all > 60 and comp_all < 60**2:
    print('Time to complete : {} min'.format(round(comp_all/60, 2)))
elif comp_all >= 60**2:
    print('Time to complete : {} h'.format(round(comp_all/(60**2), 2)))
else:
    print('Time to complete : {} s'.format(round(comp_all, 2)))



#===================================================================================================== Dict. Update ===#
print('*** Dictionary update ***')
curr_p_opt = p_opt_all[0]
curr_perr_opt = perr_all[0]

start_2 = time.time()
Hopt =  fwm_prog.optimize_SAFT_dictonary(p_opt = curr_p_opt, perr_opt = curr_perr_opt, oracle_cheating = False,
                                        deriv_wrt_tau = False)
a_opt = fwm_prog.get_single_ascan(Hopt) 
#b_opt = np.linalg.lstsq(Hopt, a_opt, rcond = None)[0] # (19.07.30) not converged...
del Hopt
stop_2 = time.time() - start_2
print('Dictionary update takes... {} s'.format(round(stop_2, 3)))
print('#==================#')



