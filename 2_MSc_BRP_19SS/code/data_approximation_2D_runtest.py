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

r"""
#=================================#
   Run test for FWM progressive
#=================================#
Goals
-----
The previous result showed that the performance of the first Taylor approximation (w.r.t. scan position, p_opt) 
became worse than the model at the wrong tracked position (p_track = p_true + err), which is not expected.
This script examines, where coding error (or calculation error) occurs by using a simplifeid measurement 
scenario.
Moreover, the performance of the first Taylor approximation w.r.t. the ToF, which is expected to be better
than the one w.r.t. the scan position, is examnied as well. 

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
        (3c) The pulse approximation w.r.t. the ToF(i.e. tau)
            tau_true = 0.5* distance(p_true, p_defect)/ c0
            tau_track = 0.5* distance(p_track, p_defect)/ c0
            tau_err = tau_track - tau_true
            H_opt2 = H(p_track) + dH/dtau* tau_err
            a_opt2 = np.dot(H_opt2, defect_vec)
    (4) Compare the estimation performance in terms of squared error of A-Scans

Assumptions 
-----------    
    ** 1 defect in the measurment specimen
    ** Positional error is within a small range, -2*wavelength... +2*wavelength

"""

#======================================================================================= Parameter Setting ===#
# Specimen params
Ntdata = 880
Ntdata = 500

Nxdata = 40
c0 = 6300* ureg.metre/ ureg.second
fS = 80* ureg.megahertz
dxdata = 0.5* ureg.millimetre
dzdata = 0.5* c0/(fS.to_base_units())
openingangle = 10
pos_defect = np.array([[6.0, 22.483125]])* ureg.millimetre # = [12*dxdata, 571*dzdata]
pos_defect = np.array([[6.0, 7.520625]])* ureg.millimetre

t0 = 4.75* 10**-6* ureg.second
t0 = 0* ureg.second

# Pulse params
fCarrier = 5* ureg.megahertz
B = 0.5 # Relative bandwidth factor w.r.t. fCarrier
tPulse = 10**-6* ureg.second
# Put parametters into dictionaries
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
pulse_params = {
        'fCarrier' : fCarrier,
        'B' : B,
        'tPulse' : tPulse,
        'fS' : fS
        }
# Stepsize for the SAFT dictionary (w.r.t. wavelenghth)
stepsize = 0.1* ureg.millimetre  #with unit
wavelength = c0/fCarrier.to_base_units() # with unit

# True scan position
p_true = np.array([5.0])* ureg.millimetre 

#================================================================================================== a_true ===#
print('Using reflectivity matrix')
fwm_prog = DataGeneratorProgressive2D(specimen_params, stepsize)
fwm_prog.unit_handling_specimen_params()
fwm_prog.vectorize_defect_map()

fwm_prog.set_reflectivity_matrix(p_true, oracle_cheating = True)
fwm_prog.set_pulse_parameter(pulse_params)
fwm_prog.compute_scipyGausspulse()
adelta = fwm_prog.get_single_ascan(fwm_prog.refmatrix)
Htrue = fwm_prog.get_impulse_response_dictionary(without_refmatrix = False)
atrue = fwm_prog.get_single_ascan(Htrue)
# Get ToF for a_opt2 calculation (-> the difference of the ToF is required )
tof_true = fwm_prog.tof_calculator.tof #[S], unitless
# Delete teh dictionary to reduce the memory consumption
del Htrue

#================================================================================================== a_track ===#
# Set p_track
error = 0.7* wavelength
p_track = p_true + error #with unit
print(p_track)
# Set the reflectivity matrix based on teh p_track
fwm_prog.set_reflectivity_matrix(p_track, oracle_cheating = True)
Htrack = fwm_prog.get_impulse_response_dictionary(without_refmatrix = False)
atrack = fwm_prog.get_single_ascan(Htrack)
# Get ToF for a_opt2 calculation (-> the difference of the ToF is required )
tof_track = fwm_prog.tof_calculator.tof #[S], unitless

#================================================================================================== a_opt1 ===#
error_x = fwm_prog.unit_handling_positrional_error(error)

start = time.time()
Hderiv = fwm_prog.get_derivative_dict(deriv_wrt_tau = False)
Hopt1 = Htrack + Hderiv* error_x
del Hderiv
aopt1 = fwm_prog.get_single_ascan(Hopt1)
del Hopt1
stop = time.time() - start
print('Derivative w.r.t. x takes... {} S'.format(stop))
# normalize the A-Scans
aopt1 = aopt1/abs(aopt1).max()

#================================================================================================== a_opt2 ===#
error_tof = tof_track - tof_true #[S], unitless -> no handling necessary!

start = time.time()
Hderiv = fwm_prog.get_derivative_dict(deriv_wrt_tau = True)
Hopt2 = Htrack - Hderiv* error_tof # -> works better with subtraction, why?????
del Hderiv
aopt2 = fwm_prog.get_single_ascan(Hopt2)
del Hopt2
stop = time.time() - start
print('Derivative w.r.t. tau takes... {} S'.format(stop))
# normalize the A-Scans
aopt2 = aopt2/abs(aopt2).max()

del Htrack

#================================================================================================= a_true2 ===#
print('Direct Gabor calculation!')
###### Using GabourPulse instead of Scipy #######
alpha = fwm_prog.alpha* (ureg.hertz)**2

del fwm_prog
pulse_params_new = {
        'fCarrier' : fCarrier,
        'fS' : fS,
        'alpha' : alpha
        }
fwm_prog = DataGeneratorProgressive2D(specimen_params, stepsize)
fwm_prog.set_pulse_parameter(pulse_params_new)
fwm_prog.unit_handling_specimen_params()
fwm_prog.vectorize_defect_map()

fwm_prog.set_reflectivity_matrix(p_true, oracle_cheating = True)

Htrue = fwm_prog.get_impulse_response_dictionary(without_refmatrix = True)
atrue2 = fwm_prog.get_single_ascan(Htrue)
# Get ToF for a_opt2 calculation (-> the difference of the ToF is required )
tof_true2 = fwm_prog.tof_calculator.tof #[S], unitless
# Delete teh dictionary to reduce the memory consumption
del Htrue

#================================================================================================= a_track2 ===#

error = 0.7* wavelength
p_track = p_true + error #with unit
print(p_track)

fwm_prog.set_reflectivity_matrix(p_track, oracle_cheating = True)
Htrack = fwm_prog.get_impulse_response_dictionary(without_refmatrix = True)
atrack2 = fwm_prog.get_single_ascan(Htrack)
# Get ToF for a_opt2 calculation (-> the difference of the ToF is required )
tof_track2 = fwm_prog.tof_calculator.tof #[S], unitless

#================================================================================================== a_opt3 ===#
error_x = fwm_prog.unit_handling_positrional_error(error)

start = time.time()
Hderiv = fwm_prog.get_derivative_dict(deriv_wrt_tau = False)
Hopt1 = Htrack + Hderiv* error_x
del Hderiv
aopt3 = fwm_prog.get_single_ascan(Hopt1)
del Hopt1
stop = time.time() - start
print('Derivative w.r.t. x takes... {} S'.format(stop))
# normalize the A-Scans
aopt3 = aopt3/abs(aopt3).max()

#================================================================================================== a_opt4 ===#
error_tof = tof_track - tof_true #[S], unitless -> no handling necessary!

start = time.time()
Hderiv = fwm_prog.get_derivative_dict(deriv_wrt_tau = True)
Hopt2 = Htrack - Hderiv* error_tof # -> works better with subtraction, why?????
del Hderiv
aopt4 = fwm_prog.get_single_ascan(Hopt2)
del Hopt2
stop = time.time() - start
print('Derivative w.r.t. tau takes... {} S'.format(stop))
# normalize the A-Scans
aopt4 = aopt4/abs(aopt4).max()

del Htrack

"""


#plt.plot(adelta)
plt.plot(atrue)
plt.plot(atrack)
plt.plot(aopt1)
plt.plot(aopt2)
plt.legend(['atrue', 'aopt1', 'aopt2'])

"""

#===================================================================================================== MSE ===#
# for my A-Scan
analyzer = ImageQualityAnalyzerMSE(np.array([atrue]))
se_track = analyzer.get_mse(np.array([atrack]))
se_track4 = analyzer.get_mse(np.array([atrack2]))
se_opt1_single = analyzer.get_mse(np.array([aopt1]))
se_opt2_single = analyzer.get_mse(np.array([aopt2]))
se_opt3_single = analyzer.get_mse(np.array([aopt3]))
se_opt4_single = analyzer.get_mse(np.array([aopt4]))



print('Performance comparision: SE')
print('Track : {}'.format(se_track))
print('Track2 : {}'.format(se_track4))
print('Opt1 : {}'.format(se_opt1_single))
print('Opt2 : {}'.format(se_opt2_single))
print('Opt3 : {}'.format(se_opt3_single))
print('Opt4 : {}'.format(se_opt4_single))



