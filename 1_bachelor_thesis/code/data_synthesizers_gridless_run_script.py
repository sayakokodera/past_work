
import numpy as np
from ultrasonic_imaging_python.forward_models import data_synthesizers_gridless 
from ultrasonic_imaging_python.definitions import units
ureg = units.ureg
from ultrasonic_imaging_python.visualization.slice_figures import SliceFigure3D

import matplotlib.pyplot as plt

plt.close('all')

###################################################################################################### Parameters ######
# constant
c0 = 6300* ureg.meter / ureg.second #5920
openingangle = 80 # better not to change this!!!!!
anglex = 0 #grad
angley = 0 #grad
Ntdata = 880
fS = 80* ureg.megahertz  
tS = 1/(80*10**6)* ureg.second
t0 = 0.000005625*ureg.second
# positions 
dx = 0.5* ureg.millimeter
dx_magn = dx.to_base_units().magnitude
dy = 0.5* ureg.millimeter
dy_magn = dy.to_base_units().magnitude
dz = (0.5* c0* tS).magnitude      
# defect positions 
defect_map = [[10, 9, 520], [16, 18, 554], [20, 23, 571], [29, 32, 614]]
pos_defect_x = np.array([10, 16, 20, 29])* dx_magn #[8, 3, 13]
pos_defect_y = np.array([9, 18, 23, 32])* dy_magn # [7, 6, 4]
pos_defect_z = np.array([520, 554, 571, 614])*dz # [3, 18, 7]
pos_defect_unitless = np.vstack((pos_defect_x, pos_defect_y, pos_defect_z))
pos_defect_unitless = pos_defect_unitless.transpose()
pos_defect = np.array(pos_defect_unitless)* ureg.meter
# scan positions 
#x_transducer_list = [0.27, 8, 2.35, 21.08, 26.32, 5.72, 13.28, 20.48, 0.2, 20.49, 2.05]
#y_transducer_list = [9.5, 6.06, 11.44, 4.86, 14.98, 12.3, 8.57, 3.96, 6.5, 6.89, 15.51]
x_list = []
y_list = []
for y in range(40):
    for x in range(40):
        x_list.append(x)
        y_list.append(y)
        

x_transducer = dx*np.array(x_list) 
y_transducer = dy*np.array(y_list) 
z_transducer = np.zeros(len(x_transducer))* ureg.meter

### parameter settings for grid data ###
specimen_parameters = {'dxdata': dx,
                       'dydata': dy,
                       'c0': c0,
                       'fS': fS,
                       'Nxdata': 40,
                       'Nydata': 40,
                       'Ntdata': Ntdata,
                       'Nxreco': 40,
                       'Nyreco': 40,
                       'Nzreco': Ntdata,
                       'openingangle': openingangle,  # degree
                       'anglex': 0,  # degree
                       'angley': 0,  # degree
                       'xImage': 0* ureg.millimeter,#10 * 0.5 * ureg.millimeter,
                       't0': t0,
                       #'pos_defect' : pos_defect,
                       'defect_map' : defect_map,
                       'x_transducer' : x_transducer, 
                       'y_transducer' : y_transducer,
                       'z_transducer' : z_transducer
                       }

pulse_parameters = {'tPulse' : 20/fS, 
                    'fCarrier' : 4* ureg.megahertz, 
                    'B' : 1, 
                    'fS' : fS
                    }

################################################################################################# Data Generation ######
fwm_gridless = data_synthesizers_gridless.GridlessDataGenerator3D()
fwm_gridless.register_parameters(specimen_parameters)
fwm_gridless.set_pulse_parameters('Gaussian', pulse_parameters)
data = fwm_gridless.calculate_bscan()  
cimg_data_sum = fwm_gridless.generate_cscan_format(data = data, summation = True,
                                                   Nxdata = specimen_parameters['Nxdata'], 
                                                   Nydata = specimen_parameters['Nydata'], 
                                                   x_scan_idx = x_list, y_scan_idx = y_list, save_data = False)
data_3D = fwm_gridless.data_3D

# get parameters
parameter_dict = fwm_gridless.get_parameter_dictionary()
# reduce size of the data
data_small = np.array(data[0:430, :])
############################################################################################################ Reco ######
from ultrasonic_imaging_python.reconstruction.saft_for_manual_scan import SAFTonGridforManualScan

compute_saft = SAFTonGridforManualScan('3DSingleMedium', enable_file_IO = True)

reco_params = dict(specimen_parameters)
reco_params.update({
        'Ntdata' : data_small.shape[0],
        'Nzreco' : data_small.shape[0]
        })
compute_saft.set_forward_model_parameters(reco_params)

reco = compute_saft.get_reco(data_small, x_list, y_list, save_data = False)
cimg_reco_sum = compute_saft.generate_cscan_format(reco, summation = True, save_data = False)


############################################################################################################ Plot ######
plt.figure(1)
plt.imshow(data)


plt.figure(2)
plt.imshow(data_small)

plt.figure(3)
plt.imshow(cimg_reco_sum)

### data presentation using SliceFigure3D ###
# defining the corresponding values of the axes
t_axis = np.flip(np.arange(0, data_3D.shape[0]), 0) / parameter_dict['fS'] + parameter_dict['t0']
x_axis = np.arange(0, data_3D.shape[1]) #* dx_magn
y_axis = np.arange(0, data_3D.shape[2]) #* dy_magn
t_label = 'time axis'
x_label = 'x axis'
y_label = 'y axis'

dz = (specimen_parameters['c0'] / (2.0 * specimen_parameters['fS']))
z_axis_reco = np.flip(np.arange(0, reco.shape[0]), 0) * dz * (-1.0) 
x_axis_reco = np.arange(0, reco.shape[1]) * specimen_parameters['dxdata']
y_axis_reco = np.arange(0, reco.shape[2]) * specimen_parameters['dydata'] 
z_label_reco = 'z axis'
x_label_reco = 'x axis'
y_label_reco = 'y axis'

fig1 = SliceFigure3D(data_3D, 2,
                     title = 'Gridless scanned raw data',
                     axis_range = [t_axis, x_axis, y_axis],
                     axis_label = [t_label, x_label, y_label],
                     initial_slice = 0,
                     even_aspect = False,
                     display_text_info=True,
                     margin = 0.12)

fig2 = SliceFigure3D(reco, 2,
                    title = 'Reconstruction',
                    axis_range = [z_axis_reco, x_axis_reco, y_axis_reco],
                    axis_label = [z_label_reco, x_label_reco, y_label_reco],
                    initial_slice = 15,
                    even_aspect = False,
                    display_text_info=True,
                    margin = 0.12)

plt.show()
