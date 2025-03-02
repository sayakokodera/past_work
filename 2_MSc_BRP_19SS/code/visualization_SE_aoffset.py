# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import time

from tools.datetime_formatter import DateTimeFormatter
from ultrasonic_imaging_python.definitions import units
ureg = units.ureg
Q_ = ureg.Quantity


from ultrasonic_imaging_python.forward_models.data_synthesizers_progressive import DataGeneratorProgressive2D
from ultrasonic_imaging_python.sako_tools.parameter_dictionary_exporter import ParameterDictionaryExporter
from ultrasonic_imaging_python.sako_tools.image_quality_analyzer import ImageQualityAnalyzerMSE


"""
Compare a_offset at different scan positions

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
alpha = (2.5* ureg.megahertz)**2 # prameter value from Jan

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
p_true_all = np.array([2.5, 5, 7.5, 9])

# Set the error range
Nsamples = 101 #-> should be odd number as the error range includes 0
err_range = np.around(np.linspace(-1, 1, Nsamples), 4)* 2* wavelength
err_rangefunc = 'np.around(np.linspace(-1, 1, Nsamples), 4)* 2* wavelength'
xvalue = err_range/wavelength    


#========================================================================================= Setting for FWM ===#
# Set the base of SE
se_all = np.zeros((len(err_range), p_true_all.shape[0] + 1))
se_all[:, 0] = xvalue
# get time
start_all = time.time()  

fwm_prog = DataGeneratorProgressive2D(specimen_params, stepsize)
fwm_prog.set_pulse_parameter(pulse_params)
fwm_prog.unit_handling_specimen_params()
fwm_prog.vectorize_defect_map()

### Iteration over p_true ###
for p_idx in range(p_true_all.shape[0]):
    p_true = np.array([p_true_all[p_idx]])* ureg.millimetre 
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
        
    
    for err_idx, curr_err in enumerate(err_range):
        print('*** Position No.{}, Iteration No.{} ***'.format(p_idx, err_idx))
        # Set p_est
        p_est = p_true + curr_err
        ################# aoffset ####################
        fwm_prog.set_reflectivity_matrix(p_est, oracle_cheating = True)
        Hoffset = fwm_prog.get_impulse_response_dictionary(without_refmatrix = True)
        aoffset = fwm_prog.get_single_ascan(Hoffset)
        del Hoffset
        # Get ToF for a_opt2 calculation (-> the difference of the ToF is required )
        tof_model = fwm_prog.tof_calculator.tof #[S], unitless
        ################# SE calculation ####################
        se_all[err_idx, p_idx+1] = analyzer.get_mse(np.array([aoffset]))
    
     
# Get time
comp_all = time.time() - start_all

print('All calculation takes...')
if comp_all > 60 and comp_all < 60**2:
    print('Time to complete : {} min'.format(round(comp_all/60, 2)))
elif comp_all >= 60**2:
    print('Time to complete : {} h'.format(round(comp_all/(60**2), 2)))
else:
    print('Time to complete : {} s'.format(round(comp_all, 2)))


plt.plot(xvalue, se_all[:, 1])
plt.plot(xvalue, se_all[:, 2])
plt.plot(xvalue, se_all[:, 3])
plt.plot(xvalue, se_all[:, 4])
plt.legend(['7.5mm away', '5mm away', '2.5mm away', '1mm away'])
plt.xlabel('Positional error / $\lambda$')
plt.ylabel('$ || \mathbf{a_{true}} - \hat \mathbf{a} ||_2 $')
plt.title('SE comparison at different scan positions')

"""
#======================================================================================== Param-Dict Export ===#
# Add wavelength to the specimen_params
specimen_params.update({
        'wavelength' : wavelength
        })
# Unithandling for specimen params
exporter = ParameterDictionaryExporter(specimen_params)
specimen_unitless = exporter.convert_dictionary()
# Unithandling for specimen params
exporter = ParameterDictionaryExporter(pulse_params)
pulse_unitless = exporter.convert_dictionary()
# Unithandling for specimen params
exporter = ParameterDictionaryExporter({
            'p_true' : p_true_all* ureg.millimetre, 
            'Nsamples' : Nsamples, 
            })
position_unitless = exporter.convert_dictionary()
position_unitless.update({
        'err_range' : err_rangefunc
        })


# Set parapeter log dict
params = {
        'specimen_params' : specimen_unitless,
        'pulse_params' : pulse_unitless,
        'positions' : position_unitless,
        'SE_plots' : {
                'Row 0' : 'xvalue = err_range/wavelength',
                'Row 1' : 'p_true = {}mm'.format(p_true_all[0]),
                'Row 2' : 'p_true = {}mm'.format(p_true_all[1]),
                'Row 3' : 'p_true = {}mm'.format(p_true_all[2]),
                'Row 4' : 'p_true = {}mm'.format(p_true_all[3])
                }
        }

### For filename ###
dtformatter = DateTimeFormatter()
curr_date = dtformatter.get_date_str()
curr_time = dtformatter.get_time_str()
fname = 'parameter_set/log/params_{}.json'.format(curr_time)
exporter.export_dictionary(fname, True, params)
"""
curr_time = '20190611_22h07m52s'
np.save('npy_data/mse/SE_offset_{}'.format(curr_time), se_all)

