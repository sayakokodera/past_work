############### run test script for data synthesizers for manual scan with position errors ###############

import numpy as np
import matplotlib.pyplot as plt
import json

from ultrasonic_imaging_python.forward_models import data_synthesizers_manual_scan
from ultrasonic_imaging_python.reconstruction.saft_for_manual_scan import SAFTonGridforManualScan
from ultrasonic_imaging_python.visualization.slice_figures import SliceFigure3D
from ultrasonic_imaging_python.visualization import c_images_arbitrarily_sampled
from ultrasonic_imaging_python.definitions import units
ureg = units.ureg
from ultrasonic_imaging_python.manual_scans.scan_position_synthesizers import ScanPositionSynthesizer

"""
Example usage to simulate manual masurement of UT
--- Data Generation ---
(1) set specimen (i.e. measurement) parameters (dict) by
    a) defining in the script
        1. get/define measurement positions (using position_maker)
        2. define pulse model name (str) and pulse parameters (dict) 
        3. call the desired subclass in the data_synthesizers_manual_scan
    b) loading json files containing required information 
        1. load json file
        2. choose a variable-set
        3. call the desired subclass in the data_synthesizers_manual_scan
        4. convert it into the dict using load_parameters_from_json()
        5. get/define measurement positions (using position_maker)

(2) set the measurement parameters in the selected subclass (set_measurement_parameters)
(3) set pulse parameters in the subclass (set_pulse_parameters) (---> in this script : skip it)
(4) get measurement data (get_measurement_data)
(5) configure error 

--- Reconstruction : gridded 3D SAFT ---
(6) define reconstruction parameters (dict)
(7) call the desired SAFT class 
(8) quantize the scan positions (quantize_scan_positions) 
(9) get reconstruction data
(10) visualize obtained data as sliced data or C-scan

"""

plt.close('all')

################################################################################################## data generation #####
###### parameter setup #####
#(1-b-1) load param_dict from json
fname_json_const = 'parameter_set/manual_scan_params_constant.json'
json_specimen_const_dict = json.loads(open(fname_json_const).read())
fname_json_var = 'parameter_set/manual_scan_params_variables.json'
json_var_dict = json.loads(open(fname_json_var).read())

###### file name setup #######
fname_log = 'log_manual_scan.log'
path = '/Users/sako5821/Desktop/git/2018_Sayako_Kodera_BA/txt_data/'
file_name_reco1 = path + 'manual_reco_without_err.txt'
file_name_data1 = path + 'manual_data_without_err.txt'
file_name_reco_cimg1 = path + 'manual_reco_cimg_without_err.txt'


# (1-b-2) setup variables
json_specimen_vaeiables = {}
var_set = {
        'dimension' : '0', # indicateing which data-set should be used
        'defect_map' : '3',
        'scan_path' : '0',
        } 
for var in var_set:
    json_specimen_vaeiables.update({
            var : json_var_dict['specimen_variables'][var][var_set[var]]
            })

# (1-b-3) 
erreddata_pos = data_synthesizers_manual_scan.DataGeneratorManualScanPositionError()

# (1-b-4) convert json file into a dict
erreddata_pos.load_parameters_from_json(json_specimen_const_dict, json_specimen_vaeiables)
    
# (1-b-5) get/define measurement positions
Npoints = 1000 
erreddata_pos.pick_scan_positions(Npoints, initial_seed_value = 0, grid_size = ureg.millimeter)

specimen_params = erreddata_pos.get_measurement_param_dict()

# (2)
erreddata_pos.set_measurement_parameters(params_from_json = True, measurement_params = None, 
                                         posscan_from_json = True, pos_scan = None)
# (4) generate / load data
#data = erreddata_pos.get_data(save_data = False)
data = np.load('npy_data/manual_scan_trajectory/data.npy')

plt.figure(1)
plt.imshow(data)

# (5)
erreddata_pos.error_configuration(err_range = 0.7* ureg.millimeter, initialize = True, with_unit = True, 
                                  seed_value = 0, sigma = 1)
# manipulate scan positions
erreddata_pos.manipulate_scan_positions()
x_scan_err = erreddata_pos.x_transducer_with_err
y_scan_err = erreddata_pos.y_transducer_with_err

################################################################################################### reconstruction #####

###### parameter setup #####

# (6) define reconstruction parameters (dict) 
reco_params = dict(specimen_params)
reco_params.update({
        'dxdata' : 1*ureg.millimeter,
        'dydata' : 1*ureg.millimeter,
        'dxreco' : 1*ureg.millimeter,
        'dyreco' : 1*ureg.millimeter,
        'Nxreco' : specimen_params['Nxdata'],
        'Nyreco' : specimen_params['Nydata'],
        'Nzreco' : specimen_params['Ntdata']
        })

# (7) call the desired SAFT class 
compute_saft = SAFTonGridforManualScan('3DSingleMedium', enable_file_IO = False)
compute_saft.set_forward_model_parameters(reco_params)

# (8) quantize the scan positions using the manipulated scan positions
x_scan_err_idx, y_scan_err_idx = compute_saft.quantize_scan_positions(x_scan_err, y_scan_err)
# get correct scan positions (index, instead of in m)
x_scan_idx = erreddata_pos.x_scan_idx
y_scan_idx = erreddata_pos.y_scan_idx
x_scan_correct = erreddata_pos.param_dict['x_transducer']
y_scan_correct = erreddata_pos.param_dict['y_transducer']

#(9)
reco_1 = compute_saft.get_reco(data, x_scan_err_idx, y_scan_err_idx, save_data = False)
#reco_2 = compute_saft.get_reco(data, x_scan_idx, y_scan_idx, save_data = False) # without error

#################################################################################################### visualization #####
# (10) C-scan visualization of reco
# sum
cimg_reco1 = compute_saft.generate_cscan_format(reco_1, summation = True, save_data = False)
#cimg_reco2 = compute_saft.generate_cscan_format(reco_2, summation = True, save_data = False) # without error

plt.figure(4)
plt.imshow(cimg_reco1)
plt.colorbar()


