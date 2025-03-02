# -*- coding: utf-8 -*-
import numpy as np
import json
import matplotlib.pyplot as plt
import time

from tools.datetime_formatter import DateTimeFormatter
from ultrasonic_imaging_python.definitions import units
ureg = units.ureg
Q_ = ureg.Quantity


from ultrasonic_imaging_python.forward_models.data_synthesizers_progressive import DataGeneratorProgressive2D
from ultrasonic_imaging_python.sako_manual_scans.error_generator import ScanPositionError
from ultrasonic_imaging_python.sako_tools.unit_adder import UnitAdderDictionary
from ultrasonic_imaging_python.sako_tools.unit_adder import UnitAdderArray
from ultrasonic_imaging_python.sako_tools.parameter_dictionary_converter import ParameterDictionaryConverter
from ultrasonic_imaging_python.sako_tools.parameter_dictionary_exporter import ParameterDictionaryExporter
from ultrasonic_imaging_python.sako_tools.image_quality_analyzer import ImageQualityAnalyzerMSE


"""
#==========================#
   2D Data Approximation
#==========================#
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


Assumptions 
-----------    
    ** 1 defect in the measurment specimen
    ** Positional error is within a small range, -2*wavelength... +2*wavelength


This run script does followings:
    **** Setting ****
    (1) Parameter setting
        -> either set directly or load from a json file
    **** Progressive FWM data calculation (i.e. single A-Scan calculation) ****
    (2) FWM setting
    (3) Calculate atrue
        - Set reflectivity matrix with p_true
        - Get Htrue
        - Get atrue using Htrue
        - Delete Htrue for faster calculation
        - Get tof_true using fwm.tof_calculato.tof for calculating aopt2
        - Call ImageQualityAnalyzer using atrue
    (4) Set the range of positional error (with unit)
    (5) Iterate over the error
        (6) Set p_est = p_true + curr_err
        (7) Calculate the modeled A-Scan based on p_est
            - Set reflectivity matrix with p_est
            - Get Hoffset
            - Get aoffset using Hmodel
            - Get tof_model using fwm.tof_calculato.tof for calculating aopt2
        (8) Calculate aopt1 (derivative w.r.t. p)
            - Unithandling for curr_err -> obtain err_x([m], unitless)
            - Get Hderiv (deriv_wrt_tau = False)
            - Hopt = Hoffset + Hderiv* err_x
            - Delete Hderiv for faster calculation
            - Get aopt1 using Hopt
            - Delete Hopt
        (9) Calculate aopt1 (derivative w.r.t. tau)
            - Set the ToF error: err_tof = tof_model - tof_true
            - Get Hderiv (deriv_wrt_tau = True)
            - Hopt = Hoffset - Hderiv* err_tof !!! subtraction !!!
            - Delete Hderiv for faster calculation
            - Get aopt2 using Hopt
            - Delete Hopt
        (10) Calculalte SE for aoffset, aopt1, aopt2

"""
plt.close('all')
#=============================================================================================== Parameter Setting ===#
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
#pos_defect = np.array([[12*dxdata.magnitude, 191*dzdata.magnitude]])* ureg.metre
#pos_defect = np.array([[12*dxdata.magnitude, 250*dzdata.magnitude]])* ureg.metre # Jan's setting
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
alpha = (2.5* ureg.megahertz)**2 # Better windowing
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
p_true = np.array([7.8])* ureg.millimetre 

#========================================================================================= Setting for FWM ===#
# get time
start_all = time.time()  

fwm_prog = DataGeneratorProgressive2D(specimen_params, stepsize)
fwm_prog.set_pulse_parameter(pulse_params)
fwm_prog.unit_handling_specimen_params()
fwm_prog.vectorize_defect_map()

#================================================================================================== a_true ===#

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

#==================================================================================== Iteration over Error ===#
# Set the error range
Nsamples = 101 #-> should be odd number as the error range includes 0
err_max_relative = 2 # relative to wavelength
err_range = np.around(np.linspace(-1, 1, Nsamples), 4)* err_max_relative* wavelength
err_rangefunc = 'np.around(np.linspace(-1, 1, Nsamples), 4)* {}* wavelength'.format(err_max_relative)
# Set the base of SE
xvalue = err_range/wavelength
se_offset = np.zeros((len(err_range), 2))
se_offset[:, 0] = np.array(xvalue)
se_opt1 = np.array(se_offset)
se_opt2 = np.array(se_offset)
se_opt3 = np.array(se_offset)


for idx, curr_err in enumerate(err_range):
    print('*** Iteration No.{} ***'.format(idx))
    # Set p_est
    p_est = p_true + curr_err
    ################# aoffset ####################
    fwm_prog.set_reflectivity_matrix(p_est, oracle_cheating = True)
    Hoffset = fwm_prog.get_impulse_response_dictionary(without_refmatrix = True)
    aoffset = fwm_prog.get_single_ascan(Hoffset)
    # Get ToF for a_opt2 calculation (-> the difference of the ToF is required )
    tof_model = fwm_prog.tof_calculator.tof #[S], unitless
    ################# aopt1 ####################
    err_x = fwm_prog.unit_handling_positrional_error(curr_err)
    start = time.time()
    Hderiv = fwm_prog.get_dictionary_derivative(deriv_wrt_tau = False)
    Hopt = Hoffset - Hderiv* err_x
    del Hderiv
    aopt1 = fwm_prog.get_single_ascan(Hopt)
    del Hopt
    stop = time.time() - start
    print('Derivative w.r.t. x takes... {} S'.format(round(stop, 3)))
    # normalize the A-Scans
    aopt1 = aopt1/abs(aopt1).max()
    ################# aopt2 ####################
    #error_tof = tof_model - tof_true #[S], unitless -> no handling necessary!
    #start = time.time()
    #Hderiv = fwm_prog.get_dictionary_derivative(deriv_wrt_tau = True)
    #Hoptsub = Hoffset - Hderiv* error_tof # -> works better with subtraction, why?????
    # Hoptadd = Hoffset + Hderiv* error_tof # -> much worse performance
    #del Hderiv
    #aopt2 = fwm_prog.get_single_ascan(Hoptsub)
    #del Hoptsub
    #stop = time.time() - start
    #print('Derivative w.r.t. tau takes... {} S'.format(round(stop, 3)))
    # normalize the A-Scans
    #aopt2 = aopt2/abs(aopt2).max()
    #aopt3 = aopt3/abs(aopt3).max()
    ################# SE calculation ####################
    se_offset[idx, 1] = analyzer.get_mse(np.array([aoffset]))
    se_opt1[idx, 1] = analyzer.get_mse(np.array([aopt1]))
    #se_opt2[idx, 1] = analyzer.get_mse(np.array([aopt2]))
    #se_opt3[idx, 1] = analyzer.get_mse(np.array([aopt3]))
    # Delete Hoffset for faster calculation
    del Hoffset
    
 
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
plt.plot(xvalue, se_offset[:, 1])
plt.plot(xvalue, se_opt1[:, 1])
#plt.plot(xvalue, se_opt2[:, 1])
#plt.plot(xvalue, se_opt3[:, 1])
plt.legend(['$\mathbf{a_{offset}}$', '$\mathbf{a_{opt}}$ w.r.t. x'])#, '$\mathbf{a_{opt}}$ w.r.t. $tau$'])
plt.xlabel('Positional error / $\lambda$')
plt.ylabel('$ || \mathbf{a_{true}} - \hat \mathbf{a} ||_2 $')
plt.title('Performance comparison: Squared Error')

# =============================================================================
# 
# #================================================================================================ Save Data ===#
# ### For filename ###
# dtformatter = DateTimeFormatter()
# curr_date = dtformatter.get_date_str()
# curr_time = dtformatter.get_time_str()
# 
# np.save('npy_data/mse/SE_offset_{}.npy'.format(curr_time), se_offset)
# np.save('npy_data/mse/SE_Approx_x_{}.npy'.format(curr_time), se_opt1)
# #np.save('npy_data/mse/SE_Approx_tau_{}.npy'.format(curr_time), se_opt2)
# 
# #======================================================================================== Param-Dict Export ===#
# # Add wavelength to the specimen_params
# specimen_params.update({
#         'wavelength' : wavelength
#         })
# # Unithandling for specimen params
# exporter = ParameterDictionaryExporter(specimen_params)
# specimen_unitless = exporter.convert_dictionary()
# # Unithandling for specimen params
# exporter = ParameterDictionaryExporter(pulse_params)
# pulse_unitless = exporter.convert_dictionary()
# # Unithandling for specimen params
# exporter = ParameterDictionaryExporter({
#             'p_true' : p_true, 
#             'Nsamples' : Nsamples, 
#             })
# position_unitless = exporter.convert_dictionary()
# position_unitless.update({
#         'err_range' : err_rangefunc
#         })
# 
# 
# # Set parapeter log dict
# params = {
#         'specimen_params' : specimen_unitless,
#         'pulse_params' : pulse_unitless,
#         'positions' : position_unitless,
#         'SE_plots' : {
#                 'xvalue' : 'err_range/wavelength',
#                 'yvalue' : 'SE'
#                 }
#         }
# 
# fname = 'parameter_set/log/params_{}.json'.format(curr_time)
# exporter.export_dictionary(fname, True, params)
# 
# 
# 
# 
# =============================================================================
