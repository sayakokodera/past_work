# -*- coding: utf-8 -*-
import numpy as np
import json
import matplotlib.pyplot as plt

from tools.datetime_formatter import DateTimeFormatter
from ultrasonic_imaging_python.definitions import units
ureg = units.ureg

from ultrasonic_imaging_python.sako_manual_scans.scan_position_maker import ScanPositionMaker2D
from ultrasonic_imaging_python.sako_tools.parameter_dictionary_converter import ParameterDictionaryConverter
# if this would be imported at first, then an KeyError with 'ultrasonic_imaging_python' occurs.... (2019.04.10)
from ultrasonic_imaging_python.sako_tools.parameter_dictionary_exporter import ParameterDictionaryExporter
from ultrasonic_imaging_python.sako_tools.defect_map_converter import DefectMapConverter2D
from ultrasonic_imaging_python.forward_models.data_synthesizers_gridless import GridlessDataGenerator2D


"""
2D synthetic data generation
"""
#=========================================================================================== (1) Parameter Setting ===#
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
            }

################### Constants ###################
# load parameters from json files
#!!! No comma at the end of each dictionary !!!! 
#-> otherwise json decoding error reading, "Expecting properly name enclosed in double quotes..."
#fname_const = '/Users/sako5821/Desktop/git/Master/RP_19SS/Code/parameter_set/params_constant.json'
fname_const = 'parameter_set/params_constant.json'
json_dict_const = json.loads(open(fname_const).read())
######## Specimen Constants ########
# call the class ParameterDictionaryConverter
param_converter = ParameterDictionaryConverter(json_dict_const['specimen_constants'], unit_dict)
# add unit
param_converter.add_unit_to_parameters()
# add dz
param_converter.add_dz()
# get dictionary
specimen_params = param_converter.output_dict

######## Pulse Constans ########
del param_converter
param_converter = ParameterDictionaryConverter(json_dict_const['pulse_constants'], unit_dict)
param_converter.add_unit_to_parameters()
# modify the dictionary
param_converter.modify_pulse_constants(specimen_params['fS'])
pulse_params = param_converter.output_dict
# get pulse model name
pulse_model_name = param_converter.get_pulse_model_name()


################### Variables ###################
# Variables = dimension(thus t0 and t_offset as well), pos_defect (or defect_map), pos_scan (or x_transducer)
# load parameters from json files
#fname_var = '/Users/sako5821/Desktop/git/Master/RP_19SS/Code/parameter_set/params_variables.json'
fname_var = 'parameter_set/params_variables.json'
json_dict_var = json.loads(open(fname_var).read())

######## Dimension ########
# choose which parameter set should be used (number)
dimension_setID = 3
# call the class ParameterDictionaryConverter
del param_converter
param_converter = ParameterDictionaryConverter(json_dict_var['dimension'][str(dimension_setID)], unit_dict)
# update dictionary
specimen_params = param_converter.update_dictionray(specimen_params)

######## t0 & t_offset ########
# ROI (roi = roi_range* dzdata)
roi_setID = 0
roi_range = json_dict_var['roi_range'][str(roi_setID)]
Ntdata = json_dict_var['dimension'][str(dimension_setID)]['Ntdata']['magnitude']
# t0 & t_offset calculation
t0_idx = specimen_params['Ntdata'] - roi_range
t0 = t0_idx / (specimen_params['fS'].to_base_units())
t_offset = t0

######## Scan Positions ########
# setup (-> update data storage of json file, when they are changed)
Nscans = 40 #v.190422
seed_value = 0
# call the class ScanPositionMaker2D
position_maker = ScanPositionMaker2D(Nscans, specimen_params['Nxdata'], seed_value)
# generate random scan position
position_maker.generate_random_scan_positions()
# get scan positions (without unit)
pos_scan = position_maker.get_scan_positions()
# sort the positions
pos_scan = np.sort(pos_scan, axis = 0)
# add unit
x_transducer = position_maker.add_unit(pos_scan, specimen_params['dxdata'])
# quantize possitions (for v.19.04.16)
stepsize_setID = 1
q = json_dict_var['stepsize'][str(stepsize_setID)]['magnitude']
x_transducer = np.floor((x_transducer/q))*q #floor can only be used for int
# remove the fractional part
x_transducer = np.around(x_transducer, 1)


######## Defect Positions ########
# choose parameter set
defmap_setID = 3
# load params from json_dict_var
defect_map = np.array(json_dict_var['defect_map'][str(defmap_setID)])
# convert defect_map into pos_defect (for FWM)
posdefect_converter = DefectMapConverter2D(defect_map, specimen_params['dxdata'], specimen_params['dzdata'])
pos_defect = posdefect_converter.get_pos_defect()


################### Update specimen_params ###################
specimen_params.update({
        'x_transducer' : x_transducer,
        'pos_defect' : pos_defect,
        't0' : t0,
        't_offset' : t_offset
        })


#=================================================================================== (2) Synthetic Data Generation ===#
# call FWM
gridlessfwm = GridlessDataGenerator2D()
gridlessfwm.register_parameters(specimen_params)
gridlessfwm.set_pulse_parameters(pulse_model_name, pulse_params)
data = gridlessfwm.calculate_bscan() 

# save data
fname = 'npy_data/data_2D_paramset_{}{}{}{}.npy'.format(dimension_setID, roi_setID, defmap_setID, stepsize_setID)
np.save(fname, data)

#======================================================================================================== (3) Plot ===#
fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(data)
ax.set_aspect(aspect = 0.08) # aspect ratio : aspect = 0.08


#============================================================================================= (4) Save Parameters ===#
dtformatter = DateTimeFormatter()
curr_time = dtformatter.get_time_str()
######## Specimen Parameter ########
# call the class for dictionary export
exporter = ParameterDictionaryExporter(specimen_params)
exp_dict_specimen = exporter.convert_dictionary()
# export
fname_exp_spec = 'parameter_set/log/specimen_{}.json'.format(curr_time)
exporter.export_dictionary(fname_exp_spec, False)
######## Pulse Parameter ########
del exporter
# conbine puöse_params and pulse_model_name
# call the class
exporter = ParameterDictionaryExporter(pulse_params)
exp_dict_pulse_wihtout_modelname = exporter.convert_dictionary()
exp_dict_pulse = {
        'parameters' : exp_dict_pulse_wihtout_modelname,
        'model_name' : pulse_model_name
        }
# tPulse modification
exp_dict_pulse['parameters']['tPulse'].update({
        'magnitude' : exp_dict_pulse['parameters']['tPulse']['magnitude']* 10**(-6),
        'unit' : str(ureg.second)
        })
# export
fname_exp_pulse = 'parameter_set/log/pulse_{}.json'.format(curr_time)
exporter.export_dictionary(fname_exp_pulse, True, exp_dict_pulse)



