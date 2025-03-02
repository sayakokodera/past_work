import numpy as np
import matplotlib.pyplot

from ultrasonic_imaging_python.forward_models.data_synthesizers import DataGeneratorForwardModel3D
from ultrasonic_imaging_python.reconstruction.saft_grided_algorithms import SaftEngineProgressiveUpdate3D
from ultrasonic_imaging_python.definitions.units import ureg
from ultrasonic_imaging_python.visualization.slice_figures import SliceFigure3D
from ultrasonic_imaging_python.utils.progress_bars import TextProgressBar


############################################################################################## Part 0: Parameters: #####
# load a json file containing the specimen parameters or define them in your script
specimen_parameters = {'dxdata': 0.5 * ureg.millimeter,
                       'dydata': 0.5 * ureg.millimeter,
                       'c0': 6300 * ureg.meter / ureg.second,#5920
                       'fS': 80 * ureg.megahertz,
                       'Nxdata': 40,
                       'Nydata': 40,
                       'Ntdata': 880,
                       'Nxreco': 40,
                       'Nyreco': 40,
                       'Nzreco': 880,
                       'openingangle': 20,#60, # degree
                       'anglex': 0, # degree
                       'angley': 0, # degree
                       'zImage': 0 * ureg.meter,
                       'xImage': 0 * ureg.meter,
                       'yImage': 0 * ureg.meter,
                       't0': 5.625* 10**-6 * ureg.second #0
                       }

########################################################################################### Part 1: Generate data: #####

# load a json file containing the pulse parameters or define them in your script
pulse_parameters = {'tPulse': 20 / specimen_parameters['fS'],
                    'fCarrier': 5 * ureg.megahertz,
                    'fS': 80 * ureg.megahertz,
                    'B': 0.5}

# define a map which defines where we want to see reflectors in our synthetic measurement data
defect_map = np.zeros([specimen_parameters['Nzreco'], specimen_parameters['Nxreco'], specimen_parameters['Nyreco']],
                      dtype = np.float32)
defects = [[10, 9, 520], [16, 18, 554], [20, 23, 571], [29, 32, 614]] #x, y, z
defect_map[520, 10, 9] = 1
defect_map[554, 16, 18] = 1
defect_map[571, 20, 23] = 1
defect_map[614, 29, 32] = 1

# initialize 3D data generator and set parameters
data_generator_3D = DataGeneratorForwardModel3D('3DSingleMedium', 'Gaussian')
data_generator_3D.set_forward_model_parameters(specimen_parameters)
data_generator_3D.set_pulse_parameters(pulse_parameters)
data_generator_3D.set_defect_map(defect_map[:,:])
progress_bar_data = TextProgressBar(50, 'computing data')
data = data_generator_3D.get_data(progress_bar = progress_bar_data)
progress_bar_data.finalize()

# store data to disk
np.save('npy_data/DataSynthesizers/data_grid_05.npy', data)

############################################################################################## Part 1/2: load data #####
#data = np.load('data.npy')
#data = data.copy(order = 'F')
############################################################################################# Part 2: compute SAFT #####
saft_processor = SaftEngineProgressiveUpdate3D('3DSingleMedium', enable_file_IO = True)
saft_processor.set_forward_model_parameters(specimen_parameters)

# allocate reconstruction data field
reco = np.zeros((specimen_parameters['Nzreco'], specimen_parameters['Nxreco'], specimen_parameters['Nyreco']),
                 dtype = np.float32)

progress_bar_reco = TextProgressBar(50, 'reconstructed A-Scans',
                                   specimen_parameters['Nxdata'] * specimen_parameters['Nydata'], 1)

# iterate over the single A-scans and add the to the reconstruction
for idx_x in range(specimen_parameters['Nxdata']):
    for idx_y in range(specimen_parameters['Nydata']):
        a_scan = data[:, idx_x, idx_y]
        reco = reco + saft_processor.get_reconstruction_ascan(idx_x, idx_y, a_scan)
        progress_bar_reco.increment()
progress_bar_reco.finalize()

######################################################################################### Part 3: visualize result #####
# defining the corresponding values of the axes
t_axis_data = np.flip(np.arange(0, data.shape[0]),0) / specimen_parameters['fS'] + specimen_parameters['t0']
x_axis_data = np.arange(0, data.shape[1]) * specimen_parameters['dxdata']
y_axis_data = np.arange(0, data.shape[2]) * specimen_parameters['dydata']
t_label_data = 'time axis'
x_label_data = 'x axis'
y_label_data = 'y axis'

dz = (specimen_parameters['c0'] / (2.0 * specimen_parameters['fS']))
z_axis_reco = np.flip(np.arange(0, reco.shape[0]), 0) * dz * (-1.0) + specimen_parameters['zImage']
x_axis_reco = np.arange(0, reco.shape[1]) * specimen_parameters['dxdata'] + specimen_parameters['xImage']
y_axis_reco = np.arange(0, reco.shape[2]) * specimen_parameters['dydata'] + specimen_parameters['yImage']
z_label_reco = 'z axis'
x_label_reco = 'x axis'
y_label_reco = 'y axis'

fig1 = SliceFigure3D(data, 2,
                    title = 'Synthetic data',
                    axis_range = [t_axis_data, x_axis_data, y_axis_data],
                    axis_label = [t_label_data, x_label_data, y_label_data],
                    initial_slice = 15,
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

matplotlib.pyplot.show()


np.save('npy_data/DataSynthesizers/reco_grid_05.npy', reco)

#######################################################################################################################
# =============================================================================
# # NEW 14.09.18 to test SAFTforManualScan should be deleted afterwards!!!!
# copy_data = np.array(data)
# arr1 = copy_data.transpose(2, 0, 1)
# arr2 = arr1.reshape(specimen_parameters['Ntdata'], specimen_parameters['Nxdata']* specimen_parameters['Nydata'])
# 
# pos_scan = []
# for y in range(specimen_parameters['Nydata']):
#     for x in range(specimen_parameters['Nxdata']):
#         pos_scan.append([x, y])
# pos_scan = np.array(pos_scan)
# x_scan, y_scan = list(pos_scan[:, 0]), list(pos_scan[:, 1])
# =============================================================================
