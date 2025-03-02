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

"""
#=================================#
   Iterative Dictionary Update
#=================================#
Scenario
--------
    (1) An A-Scan, a_true, is taken at the position p_true
    (2) The tracking camera recognized p_true as p_track(= p_true + p_err)
    (3) The a_true is estimated with the knwon defect position(= defect_vec) in following ways:
        (3a) The pulse model at p_track
            a_track = np.dot(H(p_track), defect_vec)
        (3b) The pulse approximation w.r.t. p_track
            p_err = p_track - p_true
            H_opt1 = H(p_track) + dH/dp_track* p_err
            a_opt1 = np.dot(H_opt1, defect_vec)


Assumptions 
-----------    
    ** 1 defect in the measurment specimen
    ** Positional error is within a small range, -2*wavelength... +2*wavelength

"""
#plt.close('all')
#================================================================================================ Function ===#
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
        
        if se_min <= 0.01:
            break
        
        elif curr_p_est < 0:
            break
    
    print("Final error:", round(se_min, 5))
    print('Final scan position: {}mm'.format(round(p_est* 10**3, 3)))
    return p_est, aopt, se_min


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
p_true_all = np.array([8.5])#np.array([2.5, 5, 7.5, 9, 9.37])

# Set the error range
Nsamples = 101 #-> should be odd number as the error range includes 0
err_max_relative = 2 # relative to teh wavelength
err_range = np.around(np.linspace(-1, 1, Nsamples), 4)* err_max_relative* wavelength
#err_range = np.array([0.38, 0.4, 0.8])* err_max_relative* wavelength
err_rangefunc = 'np.around(np.linspace(-1, 1, Nsamples), 4)* {}* wavelength'.format(err_max_relative)
err_max = 5* wavelength.magnitude

# Choose Niteration for SGD
Niteration = 10

#===================================================================================================== SGD ===#

# Set the base of SE & errx_corrected -> to store the info
xvalue = err_range/wavelength

se = np.zeros((len(err_range), len(p_true_all)+1 ))
se[:, 0] = np.array(xvalue)
errx_corrected = np.array(se)

# Get time
start_all = time.time() 

fwm_prog = DataGeneratorProgressive2D(specimen_params, stepsize)
fwm_prog.set_pulse_parameter(pulse_params)
fwm_prog.unit_handling_specimen_params()
fwm_prog.vectorize_defect_map()

### Iteration over p_true ###
for p_idx in range(p_true_all.shape[0]):
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

    for idx, curr_err in enumerate(err_range):
        print('*** Position No.{}, Error No.{} ***'.format(p_idx, idx))
        # Set p_track
        p_track = ((p_true + curr_err).to_base_units()).magnitude
        print('Current error: {}'.format(curr_err))
    
        curr_pest, aopt, curr_se = minimize_error_with_SGD(atrue, p_track, fwm_prog, analyzer, Niteration,
                                                           err_max)
        se[idx, p_idx + 1] = curr_se
        errx_corrected[idx, p_idx + 1] = (curr_pest - p_true_unitless)/ (wavelength.magnitude)
    
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

#===================================================================================================== Plot ===#
"""
plt.figure(1)
plt.plot(xvalue, se[:, 4], label = '1mm away')
plt.plot(xvalue, se[:, 3], '--', label = '2.5mm away')
plt.plot(xvalue, se[:, 2], '--', label = '5mm away')
plt.plot(xvalue, se[:, 1], '--', label = '7.5mm away')
plt.legend()
plt.xlabel('Positional error / $\lambda$')
plt.ylabel('$ || \mathbf{a_{true}} - \hat \mathbf{a} ||_2 $')
plt.title('Performance : squared error')

plt.figure(2)
plt.plot(xvalue, errx_corrected[:, 4], label = '1mm away')
plt.plot(xvalue, errx_corrected[:, 3], '--', label = '2.5mm away')
plt.plot(xvalue, errx_corrected[:, 2], '--', label = '5mm away')
plt.plot(xvalue, errx_corrected[:, 1], '--', label = '7.5mm away')
plt.legend()
plt.xlabel('$\mathbf{p_{track}} - \mathbf{p_{true}}$ / $\lambda$')
plt.ylabel('$\mathbf{p_{opt}} - \mathbf{p_{true}}$ / $\lambda$')
#plt.ylim(-10, 10)
plt.title('Performance: error correction')
"""
plt.figure(1)
plt.plot(xvalue, se[:, 1], label = '{} $\lambda$ away'.format(round((10 - p_true_all[0])/wvl_magmm, 1)))
plt.legend()
plt.xlabel('Positional error / $\lambda$')
plt.ylabel('$ || \mathbf{a_{true}} - \hat \mathbf{a} ||_2 $')
plt.ylim(0, 1)
plt.title('Performance : squared error')

plt.figure(2)
plt.plot(xvalue, errx_corrected[:, 1], label = '{} $\lambda$ away'.format(round((10 - p_true_all[0])/wvl_magmm, 1)))
plt.legend()
plt.xlabel('$\mathbf{p_{track}} - \mathbf{p_{true}}$ / $\lambda$')
plt.ylabel('$\mathbf{p_{opt}} - \mathbf{p_{true}}$ / $\lambda$')
#plt.ylim(-10, 10)
plt.title('Performance: error correction')



# =============================================================================
# #================================================================================================ Save Data ===#
# ### For filename ###
# dtformatter = DateTimeFormatter()
# curr_date = dtformatter.get_date_str()
# curr_time = dtformatter.get_time_str()
# 
# np.save('npy_data/mse/GD_SE_{}.npy'.format(curr_time), se)
# np.save('npy_data/mse/GD_PosErr_{}.npy'.format(curr_time), errx_corrected)
# 
# #======================================================================================== Param-Dict Export ===#
# # Add wavelength to the specimen_params
# specimen_params.update({
#         'wavelength' : wavelength
#         })
# # Unithandling for specimen params
# exporter = ParameterDictionaryExporter(specimen_params)
# specimen_unitless = exporter.convert_dictionary()
# # Unithandling for pulse params
# exporter = ParameterDictionaryExporter(pulse_params)
# pulse_unitless = exporter.convert_dictionary()
# # Unithandling for position setting
# exporter = ParameterDictionaryExporter({
#             'p_true' : p_true_all* ureg.millimetre, 
#             'Nsamples' : Nsamples, 
#             })
# position_unitless = exporter.convert_dictionary()
# position_unitless.update({
#         'err_range' : err_rangefunc
#         })
# 
# # Set parapeter log dict
# params = {
#         'specimen_params' : specimen_unitless,
#         'pulse_params' : pulse_unitless,
#         'positions' : position_unitless,
#         'SGD' : {
#                 'Niteration' : Niteration,
#                 'err_limit' : "2* wavelength"
#                 },
#         'SE_plots' : {
#                 'Row 0' : 'xvalue = err_range/wavelength',
#                 'Row 1' : 'p_true = {}mm'.format(p_true_all[0]),
#                 'Row 2' : 'p_true = {}mm'.format(p_true_all[1]),
#                 'Row 3' : 'p_true = {}mm'.format(p_true_all[2]),
#                 'Row 4' : 'p_true = {}mm'.format(p_true_all[3]),
#                 'Row 5' : 'p_true = {}mm'.format(p_true_all[4])
#                 }
#         }
# 
# fname = 'parameter_set/log/params_{}.json'.format(curr_time)
# exporter.export_dictionary(fname, True, params)
# 
# 
# =============================================================================
