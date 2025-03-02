
import numpy as np
import matplotlib.pyplot as plt

from ultrasonic_imaging_python.forward_models import data_synthesizers_gridless
from ultrasonic_imaging_python.reconstruction.saft_for_manual_scan import SAFTonGridforManualScan 

from ultrasonic_imaging_python.visualization.slice_figures import SliceFigure3D
import tools.txt_file_writer as fwriter
import tools.txt_file_reader as freader
from ultrasonic_imaging_python.definitions import units
ureg = units.ureg


plt.close('all')

###################################################################################################### Parameters ######
specimen_parameters = {'dxdata': 0.5 * ureg.millimeter,
                       'dydata': 0.5 * ureg.millimeter,
                       'c0': 5920 * ureg.meter / ureg.second,
                       'fS': 80 * ureg.megahertz,
                       'Nxdata': 30,
                       'Nydata': 20,
                       'Ntdata': 50,
                       'Nxreco': 30,
                       'Nyreco': 20,
                       'Nzreco': 50,
                       'openingangle': 80, # degree
                       'anglex': 0, # degree
                       'angley': 0, # degree
                       'zImage': 0 * ureg.meter,
                       'xImage': 0 * ureg.meter,
                       'yImage': 0 * ureg.meter,
                       't0': 0 * ureg.second
                       }


pulse_parameters = {'tPulse': 20 / specimen_parameters['fS'],
                    'fCarrier': 5 * ureg.megahertz,
                    'fS': 80 * ureg.megahertz,
                    'B': 0.5}
# defect positions
x_def = (15* specimen_parameters['dxdata']).to_base_units().magnitude
y_def = (12* specimen_parameters['dydata']).to_base_units().magnitude
# dz
tS = (1 / ((specimen_parameters['fS'].to_base_units()).magnitude))* ureg.second
dz = 0.5* specimen_parameters['c0']* tS
z_def = 20* dz.magnitude
pos_defect = np.array([x_def, y_def, z_def])* ureg.meter
# dictionary update
specimen_parameters.update({
        'pos_defect' : pos_defect
        })

# scan positions
xscan1 = []
yscan1 = []
xscan2 = []
yscan2 = []
# grid size = 0.5mm
for y1 in range(specimen_parameters['Nydata']):
    for x1 in range(specimen_parameters['Nxdata']):
        xscan1.append(x1)
        yscan1.append(y1)

# grid size = 2mm
for y2 in range(0, specimen_parameters['Nydata'], 4):
    for x2 in range(0, specimen_parameters['Nxdata'], 4):
        xscan2.append(x2)
        yscan2.append(y2)
    
################################################################################################# Data Generation ######
# grid = 0.5mm
specimen_parameters.update({
        'x_transducer' : xscan1, 
        'y_transducer' : yscan1,
        'z_transducer' : np.zeros(len(xscan1))
        })
fwm_gridless = data_synthesizers_gridless.GridlessDataGenerator3D()
fwm_gridless.register_parameters(specimen_parameters)
fwm_gridless.set_pulse_parameters('Gaussian', pulse_parameters)
data1 = fwm_gridless.calculate_bscan()  
fwriter.write_txt_file(data1, 'Ntdata, Nscan', 'test_ese_grid_data_05.txt')

# grid = 2mm
specimen_parameters.update({
        'x_transducer' : xscan2, 
        'y_transducer' : yscan2,
        'z_transducer' : np.zeros(len(xscan2))
        })
fwm_gridless = data_synthesizers_gridless.GridlessDataGenerator3D()
fwm_gridless.register_parameters(specimen_parameters)
fwm_gridless.set_pulse_parameters('Gaussian', pulse_parameters)
data2 = fwm_gridless.calculate_bscan()  
fwriter.write_txt_file(data2, 'Ntdata, Nscan', 'test_ese_grid_data_2.txt')

################################################################################################## Reconstruction ######
compute_saft = SAFTonGridforManualScan('3DSingleMedium', enable_file_IO = False)




