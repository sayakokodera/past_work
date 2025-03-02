# -*- coding: utf-8 -*-
import numpy as np
import json
import matplotlib.pyplot as plt
import time

from tools.datetime_formatter import DateTimeFormatter
from ultrasonic_imaging_python.definitions import units
ureg = units.ureg

from ultrasonic_imaging_python.forward_models.data_synthesizers_progressive import DataGeneratorProgressive2D
from ultrasonic_imaging_python.sako_manual_scans.error_generator import ScanPositionError
from ultrasonic_imaging_python.sako_tools.unit_adder import UnitAdderDictionary
from ultrasonic_imaging_python.sako_tools.unit_adder import UnitAdderArray
from ultrasonic_imaging_python.sako_tools.parameter_dictionary_converter import ParameterDictionaryConverter
from ultrasonic_imaging_python.sako_tools.image_quality_analyzer import ImageQualityAnalyzerMSE

"""
#=====================================#
Example Scenario 1 for Progressive FWM
#=====================================#
Scenario:
(1) An A-Scan a_true taken at the scan position p_true
(2) The tracked position p_track is -1.22mm away from the true scan position p_true
(3) Calculate a_track = np.dot(H(p_track), defect_vec)
(4) Calculate se_track = the squared error of a_true and a_track = 0.71...
(5) Lookup the pre-calculated MSE table to find all possible error delta_p(= np.array)
(6) Assume p_opt = p_track - delta_p[0]
(7) Calculate a_opt using
        H(p_opt)
        dH(p_opt)/dp_opt
        delta_p
(8) Calculate se_opt = the squared error of a_true and a_opt
(9) Compare se_opt and se_approx[delta_p[0]] which is pre-calculated
-> Question: how big the difference b/w these two ses?
"""
#============================================================================================ (0) Parameter Setting ===#
######## load parameters from the log ########
fname_spec = 'parameter_set/log/specimen_20190510_21h41m58s.json'
fname_pulse = 'parameter_set/log/pulse_20190510_21h41m58s.json' # with pulse length = 80
specimen_params_log = json.loads(open(fname_spec).read())
pulse_params_log = json.loads(open(fname_pulse).read())

######## add unit ########
# define unit_dict
unit_dict = {
            'c0' : ureg.meter / ureg.second,
            'fS' : ureg.megahertz,
            'zImage': ureg.meter,
            'xImage': ureg.meter,
            'yImage': ureg.meter,
            'fCarrier' : ureg.megahertz,
            'dxdata' : ureg.millimeter,
            'dydata' : ureg.millimeter,
            'dzdata' : ureg.millimeter,
            't0' : ureg.second,
            't_offset' : ureg.second,
            'x_transducer' : ureg.millimeter,
            'pos_defect' : ureg.millimeter,
            'tPulse' : ureg.second,
            }

### Specimen Constants ###
param_converter = ParameterDictionaryConverter(specimen_params_log, unit_dict)
param_converter.add_unit_to_parameters()
specimen_params = param_converter.output_dict

### Pulse Constans ###
del param_converter
param_converter = ParameterDictionaryConverter(pulse_params_log['parameters'], unit_dict)
param_converter.add_unit_to_parameters()
pulse_params = param_converter.output_dict

# Choose the right scan position -> in the future, from the log json dictionary?
posscan_set_ID = [10] # index of the selected scan position, p_true

# set stepsize for dictionary
stepsize = 0.1* ureg.millimetre

#=================================================================================================== (1) Get a_true ===#
# Load ref data
data_true = np.load('npy_data/data_2D_paramset_3031.npy')
a_true = data_true[:, posscan_set_ID[0]]

#================================================================================================== (2) Set p_track ===#
# Call the fwm class
fwm_prog = DataGeneratorProgressive2D(specimen_params, pulse_params, stepsize)
# Compute pulse
fwm_prog.compute_scipyGausspulse()
# Unit handling
fwm_prog.specimen_unit_handling()
# Get p_true
p_true = fwm_prog.x_transducer[posscan_set_ID[0]]
# Set p_track
err_true = round(12* fwm_prog.stepsize, 6) # = -1.2mm
p_track = np.array([[p_true + err_true]]) 

#============================================================================================ (3) Calculate a_track ===#
# Get dictionary
dict_track = fwm_prog.get_dictionary(p_track, np.array([0]), cheating = True)
# Vectorize the defect map
fwm_prog.vectorize_defect_map() 
# Get A-Scan
a_track = fwm_prog.get_single_ascan(dict_track)
a_track = a_track / abs(a_track).max()

#=========================================================================================== (4) Calculate se_track ===#
# Call the analyzer class
analyzer = ImageQualityAnalyzerMSE(np.array([a_true])) #ImageQualityAnalizer only works with arrays, not vectors
# Get squared error
se_track = analyzer.get_mse(np.array([a_track]))
print(se_track)

#======================================================================================== (5) Look up the MSE table ===#
###### Load the table ######
### File names







