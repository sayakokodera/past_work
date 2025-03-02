############### run test script for data synthesizers for manual scan with pressure errors (z_scan) ###############

import numpy as np
import matplotlib.pyplot as plt
import json

from ultrasonic_imaging_python.forward_models import data_synthesizers_manual_scan
from ultrasonic_imaging_python.reconstruction.saft_for_manual_scan import SAFTonGridforManualScan
from ultrasonic_imaging_python.visualization.slice_figures import SliceFigure3D
from ultrasonic_imaging_python.visualization import c_images_arbitrarily_sampled
from ultrasonic_imaging_python.definitions import units
ureg = units.ureg


"""
Example usage to simulate manual masurement of UT
--- Data Generation ---
(1) set specimen (i.e. measurement) parameters (dict) by
    a) defining in the script
        1. get/define measurement positions (using position_maker)
        2. define pulse model name (str) and pulse parameters (dict) 
        3. call the desired subclass in the data_synthesizers_manual_scan
        ---> set pulse parameters afterwards!!!
    b) loading json files containing required information 
        1. load json file
        2. choose a variable-set
        3. call the desired subclass in the data_synthesizers_manual_scan
        4. convert it into the dict using load_parameters_from_json()
        5. get/define measurement positions (using position_maker)

(2) configure error 
(3) set the measurement parameters in the selected subclass (set_measurement_parameters)
(4) get measurement data (get_measurement_data)

--- Reconstruction : gridded 3D SAFT ---
(5) define reconstruction parameters (dict)
(6) call the desired SAFT class 
(7) quantize the scan positions (quantize_scan_positions) 
(8) get reconstruction data
(9) visualize obtained data as sliced data or C-scan

"""

plt.close('all')

################################################################################################## data generation #####
###### parameter setup #####
#(1-b-1) load param_dict from json
fname_json_const = 'parameter_set/manual_scan_params_constant.json'
json_specimen_const_dict = json.loads(open(fname_json_const).read())
fname_json_var = 'parameter_set/manual_scan_params_variables.json'
json_var_dict = json.loads(open(fname_json_var).read())



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
erreddata_zscan = data_synthesizers_manual_scan.DataGeneratorManualScanZscanError()

# (1-b-4) convert json file into a dict
erreddata_zscan.load_parameters_from_json(json_specimen_const_dict, json_specimen_vaeiables)
    
# (1-b-5) get/define measurement positions
Npoints = 1000 
erreddata_zscan.pick_scan_positions(Npoints, initial_seed_value = 0, grid_size = ureg.millimeter)

specimen_params = erreddata_zscan.get_measurement_param_dict()

# (2)
erreddata_zscan.error_configuration(err_range = 0.7* ureg.millimeter, initialize = True, with_unit = True, 
                                  seed_value = 0, sigma = 1)

###### file name setup #######
path = 'npy_data/manual_scan_trajectory/zscan_err/'
fname_err = '07'
fname_data = "{}data_{}.npy".format(path, fname_err)
fname_reco = "{}reco_{}.npy".format(path, fname_err)
fname_cimg = "{}cimg_{}.npy".format(path, fname_err)

# (3)
erreddata_zscan.set_measurement_parameters(params_from_json = True, measurement_params = None, 
                                         posscan_from_json = True, pos_scan = None)
# (4) generate / load data
data = erreddata_zscan.get_data(save_data = False)
#data = np.load('npy_data/manual_scan_trajectory/zscan_err/______.npy')

plt.figure(1)
plt.imshow(data)

################################################################################################### reconstruction #####

###### parameter setup #####

# (5) define reconstruction parameters (dict) 
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

# (6) call the desired SAFT class 
compute_saft = SAFTonGridforManualScan('3DSingleMedium', enable_file_IO = False)
compute_saft.set_forward_model_parameters(reco_params)

# (7) quantize the scan positions (either with the data or reco class)
#x_scan = erreddata_zscan.param_dict['x_transducer']
#y_scan = erreddata_zscan.param_dict['y_transducer']
#x_scan_idx, y_scan_idx = compute_saft.quantize_scan_positions(x_scan, y_scan)
x_scan_idx = erreddata_zscan.x_scan_idx
y_scan_idx = erreddata_zscan.y_scan_idx


#(9)
reco = compute_saft.get_reco(data, x_scan_idx, y_scan_idx, save_data = False)

#################################################################################################### visualization #####
# (10) C-scan visualization of reco
# sum
cimg = compute_saft.generate_cscan_format(reco, summation = True, save_data = False)
pngname_cimg = "plots/manual_scan_trajectory/cimg_ms_zscan_err_{}.png".format(fname_err)

plt.figure(4)
plt.imshow(cimg)
plt.colorbar()
#plt.savefig(pngname_cimg)

######################################################################################################## save data #####
# =============================================================================
# np.save(fname_data, data)
# np.save(fname_reco, reco)
# np.save(fname_cimg, cimg)
# =============================================================================
