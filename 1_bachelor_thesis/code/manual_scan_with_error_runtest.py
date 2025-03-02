###################### run test script for data synthesizers for manual scan with error ###########################

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

r"""
Example usage to simulate manual masurement of UT
--- Data Generation ---
(1) define specimen (i.e. measurement) parameters (dict)
(2) get/define measurement positions
(3) define pulse model name (str) and pulse parameters (dict) 
(4) call the desired subclass in the data_synthesizers_manual_scan
(5) configure the error
(6) set the measurement parameters in the selected subclass (set_measurement_parameters)
(7) set pulse parameters in the subclass (set_pulse_parameters)
(8) get measurement data (get_measurement_data)

--- Reconstruction : gridded 3D SAFT ---
(10) define reconstruction parameters (dict)
(11) call the desired SAFT class 
(12) quantize the scan positions (quantize_scan_positions) 
(13) get reconstruction data
(14) visualize obtained data as sliced data or C-scan

"""

plt.close('all')

def remove_units(parameter_with_unit):
    return (parameter_with_unit.to_base_units()).magnitude

################################################################################################## data generation #####
###### parameter setup #####

# (1) define specimen (i.e. measurement) parameters (dict)
specimen_params = {'c0': 5920 * ureg.meter / ureg.second,
                   'fS': 80 * ureg.megahertz,
                   'Nxdata': 100, # in [mm] corresponding to the ROI of data (x-component) 
                   'Nydata': 100, # in [mm] corresponding to the ROI of data (x-component) 
                   'Ntdata': 200,
                   'openingangle': 60, # degree
                   'anglex': 0, # degree
                   'angley': 0, # dgree
                   't0' : 0.3*10**-6* ureg.second,
                   'zImage': 0 * ureg.meter,
                   'xImage': 0 * ureg.meter,
                   'yImage': 0 * ureg.meter,
                   'defect_map' : [[54, 69, 85], [22, 14, 47], [78, 86, 63]] #unitless
                   }

# setup defect position
dz = (specimen_params['c0'].to_base_units().magnitude / (2.0 * specimen_params['fS'].to_base_units().magnitude))* 10**3
pos_defect = np.array(specimen_params['defect_map'], float)
pos_defect[:, 2] = pos_defect[:, 2]* dz
# add defect positions to the parameter dictionary        
specimen_params.update({
        'pos_defect' : pos_defect* ureg.millimeter
        })


# (2) get/define measurement positions
Npoints = 50 
position_maker = ScanPositionSynthesizer(Npoints, specimen_params['Nxdata'], specimen_params['Nydata'], seed_value = 0)
# get scan positions from the drawn scan path
scan_path_img = '/Users/sako5821/Desktop/git/2018_Sayako_Kodera_BA/Code/scan_path_img/manual_scan_path_test.png'
position_maker.generate_scan_position_from_image(scan_path_img)
pos_scan = np.array(position_maker.coord_list)* ureg.millimeter


# (3) define pulse model name (str) and pulse parameters (dict) 
pulse_model_name = 'Gaussian'
pulse_parameters = {'tPulse': 20 / specimen_params['fS'],
                    'fCarrier': 5 * ureg.megahertz,
                    'fS': specimen_params['fS'],
                    'B': 0.5}



###### with position error #####
# (4)
DataGeneratorPositionError = data_synthesizers_manual_scan.DataGeneratorManualScanPositionError()
# (5)
DataGeneratorPositionError.error_configuration(0.7* ureg.millimeter, initialize = True, with_unit = True,
                                               seed_value = 0, sigma = 0.3)
# (6)
DataGeneratorPositionError.set_measurement_parameters(params_from_json = False, measurement_params = specimen_params, 
                                                      posscan_from_json = False, pos_scan = pos_scan)
# (7)
DataGeneratorPositionError.set_pulse_parameters(pulse_model_name, pulse_parameters)
# (8)
measurement_data_2 = DataGeneratorPositionError.get_data(save_data = False)
# option
pos_scan_err_1 = DataGeneratorPositionError.get_scan_positions_for_data_generation()

plt.figure(2)
plt.imshow(measurement_data_2)

# =============================================================================
# ###### with pressure error #####
# # (4)
# DataGeneratorPressureError = data_synthesizers_manual_scan.DataGeneratorManualContactPressureError()
# # (5)
# DataGeneratorPressureError.error_configuration(0.5* ureg.millimeter, initialize = True, with_unit = True,
#                                                seed_value = 0, sigma = 0.3)
# # (6)
# DataGeneratorPressureError.set_measurement_parameters(specimen_params, pos_scan)
# # (7)
# DataGeneratorPressureError.set_pulse_parameters(pulse_model_name, pulse_parameters)
# # (8)
# measurement_data_3 = DataGeneratorPressureError.get_measurement_data(save_data = False)
# # option
# pos_scan_err_2 = DataGeneratorPressureError.get_scan_positions_for_data_generation()
# 
# plt.figure(3)
# plt.imshow(measurement_data_3)
# 
# =============================================================================


################################################################################################### reconstruction #####

###### parameter setup #####

# (10) define reconstruction parameters (dict) 
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

# (11) call the desired SAFT class 
compute_saft = SAFTonGridforManualScan('3DSingleMedium', enable_file_IO = False)
compute_saft.set_forward_model_parameters(reco_params)
 

###### name setup #######
path = '/Users/sako5821/Desktop/git/2018_Sayako_Kodera_BA/txt_data/'
file_name_reco1 = path + 'manual_reco_without_err_2.txt'
file_name_reco2 = path + 'manual_reco_pos_err_1.txt'
file_name_reco3 = path + 'manual_reco_pressure_err.txt'


###### with position error #####
# (12)
x_scan_on_grid_2, y_scan_on_grid_2 = DataGeneratorPositionError.quantize_scan_positions(reco_params['dxdata'], 
                                                                                        reco_params['dydata'],
                                                                                        with_unit = True)
# (13)
reco_2 = compute_saft.get_reco(measurement_data_2, x_scan_on_grid_2, y_scan_on_grid_2, save_data = False)


# =============================================================================
# ###### with pressure error #####
# # (12)
# x_scan_on_grid_3, y_scan_on_grid_3 = DataGeneratorPressureError.quantize_scan_positions(reco_params['dxdata'], 
#                                                                                         reco_params['dydata'],
#                                                                                         with_unit = True)
# 
# #(13)
# reco_3 = compute_saft.get_reco(measurement_data_3, x_scan_on_grid_3, y_scan_on_grid_3, save_data = False)
# 
# 
# =============================================================================

#################################################################################################### visualization #####
# (14) C-scan visualization of reco
#### name setup ####
file_name_reco1_cimg = path + 'manual_reco_cimg_without_err_2.txt'
file_name_reco2_cimg = path + 'manual_reco_cimg_pos_err_1.txt'
file_name_reco3_cimg = path + 'manual_reco_cimg_pressure_err.txt'



# =============================================================================
# ### without error ###
# # max
# #cimg_reco1 = compute_saft.generate_cscan_format(reco_1, summation = False, save_data = False)
# # sum
# cimg_reco1 = compute_saft.generate_cscan_format(reco_1, summation = True, save_data = True,
#                                                 file_name = file_name_reco1_cimg)
# plt.figure(4)
# plt.imshow(cimg_reco1)
# plt.colorbar()
# #plt.savefig('plots/cimage_without_err_max_1.png')
# #plt.savefig('plots/cimage_without_err_sum_1.png')
# =============================================================================

### with position error ###
# max
#cimg_reco2 = compute_saft.generate_cscan_format(reco_2, summation = False, save_data = False)
# sum
cimg_reco2 = compute_saft.generate_cscan_format(reco_2, summation = True, save_data = False)
plt.figure(5)
plt.imshow(cimg_reco2)
plt.colorbar()
#plt.savefig('plots/cimage_pos_err_max_1.png')
#plt.savefig('plots/cimage_pos_err_sum_1.png')

# =============================================================================
# ### with contact pressure error ###
# # max
# cimg_reco3 = compute_saft.generate_cscan_format(reco_3, summation = False, save_data = False)
# # sum
# #cimg_reco3 = compute_saft.generate_cscan_format(reco_3, summation = True, save_data = False)
# plt.figure(6)
# plt.imshow(cimg_reco3)
# plt.colorbar()
# #plt.savefig('plots/cimage_pressure_err_max_1.png')
# #plt.savefig('plots/cimage_pressure_err_sum_1.png')
# =============================================================================

