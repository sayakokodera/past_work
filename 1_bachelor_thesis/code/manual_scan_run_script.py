###################### run test script for data synthesizers for manual scan ###########################

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
 
(2) configure error (---> in this script : skip it)
(3) set the measurement parameters in the selected subclass (set_measurement_parameters)
(4) set pulse parameters in the subclass (set_pulse_parameters) (---> in this script : skip it)
(5) get measurement data (get_measurement_data)

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
        'defect_map' : '0',
        'scan_path' : '0',
        } 
for var in var_set:
    json_specimen_vaeiables.update({
            var : json_var_dict['specimen_variables'][var][var_set[var]]
            })

# (1-b-3) 
DataGeneratorWithoutError = data_synthesizers_manual_scan.DataGeneratorManualScanWithoutError()

# (1-b-4) convert json file into a dict
DataGeneratorWithoutError.load_parameters_from_json(json_specimen_const_dict, json_specimen_vaeiables)
    
# (1-b-5) get/define measurement positions
Npoints = 50 
DataGeneratorWithoutError.pick_scan_positions(Npoints, initial_seed_value = 0)

specimen_params = DataGeneratorWithoutError.get_measurement_param_dict()

# (3)
DataGeneratorWithoutError.set_measurement_parameters(params_from_json = True, measurement_params = None, 
                                                     posscan_from_json = True, pos_scan = None)

# (5)
data = DataGeneratorWithoutError.get_data(save_data = False)

plt.figure(1)
plt.imshow(data)

# get scan positions
pos_scan = DataGeneratorWithoutError.get_scan_positions()

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

###### without error #####
# (8)
x_scan_on_grid, y_scan_on_grid = compute_saft.quantize_scan_positions(pos_scan['x_transducer'],
                                                                          pos_scan['y_transducer'])
#(9)
reco_1 = compute_saft.get_reco(data, x_scan_on_grid, y_scan_on_grid, save_data = False)


#################################################################################################### visualization #####
# (10) C-scan visualization of reco
### without error ###
# max
#cimg_reco1 = compute_saft.generate_cscan_format(reco_1, summation = False, save_data = False)
# sum
cimg_reco1 = compute_saft.generate_cscan_format(reco_1, summation = True, save_data = False)
plt.figure(4)
plt.imshow(cimg_reco1)
plt.colorbar()
#plt.savefig('plots/cimage_without_err_max_1.png')
#plt.savefig('plots/cimage_without_err_sum_1.png')



