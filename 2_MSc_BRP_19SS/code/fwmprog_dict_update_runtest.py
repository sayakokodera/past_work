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
#===============================================#
   Run-test : FWM Iterative Dictionary Update
#===============================================#
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
print('Defect position : {}'.format(pos_defect.to(ureg.millimetre)))

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

# True scan position
p_true_all = np.array([9])#np.array([2.5, 5, 7.5, 9])

# Set the error range
Nsamples = 101 #-> should be odd number as the error range includes 0
perr_max = 2* wavelength.magnitude
perr_range = np.around(np.linspace(-1, 1, Nsamples), 4)* 2* wavelength
perr_rangefunc = 'np.around(np.linspace(-1, 1, Nsamples), 4)* 2* wavelength'


# Choose Niteration for SGD
Niteration = 10
# Choose the targeted value for squared error
se_target = 0.01

#===================================================================================================== SGD ===#

# Set the base of SE & errx_corrected -> to store the info
xvalue = perr_range/wavelength

se = np.zeros((len(perr_range), len(p_true_all)+1 ))
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
    start_allp = time.time()
    
    p_true = np.array([p_true_all[p_idx]])* ureg.millimetre 
    fwm_prog.set_reflectivity_matrix(p_true, oracle_cheating = True)
    a_delta = fwm_prog.get_single_ascan(fwm_prog.refmatrix)
    Htrue = fwm_prog.get_impulse_response_dictionary(without_refmatrix = True)
    a_true = fwm_prog.get_single_ascan(Htrue)
    # Delete teh dictionary to reduce the memory consumption
    del Htrue
    # Get p_true without unit -> for evaulating the error correction
    p_true_unitless = fwm_prog.x_scan

    for idx, curr_err in enumerate(perr_range):
        # Get time
        start_1 = time.time()
        print('#==================#')
        print('*** Position No.{}, Error No.{} ***'.format(p_idx, idx))
        # Set p_track
        p_track = ((p_true + curr_err).to_base_units()).magnitude
        # Set the max. range of possitional error
        curr_perrmax = curr_err.magnitude
        print('Error = {}'.format(curr_err.to(ureg.millimetre)))
        
        # Minimize the positional error 
        curr_p_opt, curr_perr_opt, curr_se = fwm_prog.minimize_positional_error(a_true, p_track, perr_max, se_target, 
                                                                                Niteration, ret_semin = True)
        se[idx, p_idx + 1] = curr_se
        errx_corrected[idx, p_idx + 1] = (curr_p_opt - p_true_unitless)/ (wavelength.magnitude)
        # Get time
        stop_1 = time.time() - start_1
        print('*********')
        print('Single GD takes... {} s'.format(round(stop_1, 3)))
        print('*********')
        
    # Get time
    stop_allp = time.time() - start_allp
    print('#==================#')
    print('Calculation for Position No.{} takes...'.format(p_idx))
    print('{} s'.format(round(stop_allp, 3)))
    print('#==================#')
# =============================================================================
# 
# # Dictionary update
# print('*** Dictionary update ***')
# start_2 = time.time()
# Hopt =  fwm_prog.optimize_SAFT_dictonary(p_opt = curr_p_opt, perr_opt = curr_perr_opt,oracle_cheating = False,
#                                          deriv_wrt_tau = False)
# a_opt = fwm_prog.get_single_ascan(Hopt) 
# b_opt = np.linalg.lstsq(Hopt, a_opt, rcond = None)[0] # (19.07.30) not converged...
# del Hopt
# stop_2 = time.time() - start_2
# print('Dictionary update takes... {} s'.format(round(stop_2, 3)))
# print('#==================#')
# =============================================================================

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
plt.plot(xvalue, se[:, 1], label = '{}mm away'.format(10 - p_true_all[0]))
plt.legend()
plt.xlabel('Positional error / $\lambda$')
plt.ylabel('$ || \mathbf{a_{true}} - \hat \mathbf{a} ||_2 $')
plt.ylim(0, 1)
plt.title('Performance : squared error')

plt.figure(2)
plt.plot(xvalue, errx_corrected[:, 1], label = '{}mm away'.format(10 - p_true_all[0]))
plt.legend()
plt.xlabel('$\mathbf{p_{track}} - \mathbf{p_{true}}$ / $\lambda$')
plt.ylabel('$\mathbf{p_{opt}} - \mathbf{p_{true}}$ / $\lambda$')
#plt.ylim(-10, 10)
plt.title('Performance: error correction')


"""
#================================================================================================ Save Data ===#
### For filename ###
dtformatter = DateTimeFormatter()
curr_date = dtformatter.get_date_str()
curr_time = dtformatter.get_time_str()

np.save('npy_data/mse/SE_SGD_{}.npy'.format(curr_time), se)
np.save('npy_data/mse/PosErr_SGD_{}.npy'.format(curr_time), errx_corrected)

#======================================================================================== Param-Dict Export ===#
# Add wavelength to the specimen_params
specimen_params.update({
        'wavelength' : wavelength
        })
# Unithandling for specimen params
exporter = ParameterDictionaryExporter(specimen_params)
specimen_unitless = exporter.convert_dictionary()
# Unithandling for pulse params
exporter = ParameterDictionaryExporter(pulse_params)
pulse_unitless = exporter.convert_dictionary()
# Unithandling for position setting
exporter = ParameterDictionaryExporter({
            'p_true' : p_true_all* ureg.millimetre, 
            'Nsamples' : Nsamples, 
            })
position_unitless = exporter.convert_dictionary()
position_unitless.update({
        'err_range' : perr_rangefunc
        })

# Set parapeter log dict
params = {
        'specimen_params' : specimen_unitless,
        'pulse_params' : pulse_unitless,
        'positions' : position_unitless,
        'SGD' : {
                'Niteration' : Niteration,
                'err_limit' : "2* wavelength"
                },
        'SE_plots' : {
                'Row 0' : 'xvalue = err_range/wavelength',
                'Row 1' : 'p_true = {}mm'.format(p_true_all[0]),
                'Row 2' : 'p_true = {}mm'.format(p_true_all[1]),
                'Row 3' : 'p_true = {}mm'.format(p_true_all[2]),
                'Row 4' : 'p_true = {}mm'.format(p_true_all[3])
                }
        }

fname = 'parameter_set/log/params_{}.json'.format(curr_time)
exporter.export_dictionary(fname, True, params)
"""