# -*- coding: utf-8 -*-
import numpy as np
import json
import matplotlib.pyplot as plt
import time

from tools.datetime_formatter import DateTimeFormatter
from ultrasonic_imaging_python.definitions import units
ureg = units.ureg

from ultrasonic_imaging_python.forward_models.data_synthesizers_progressive import DataGeneratorProgressive2D
from ultrasonic_imaging_python.sako_tools.image_quality_analyzer import ImageQualityAnalyzerMSE
from ultrasonic_imaging_python.sako_tools.parameter_dictionary_exporter import ParameterDictionaryExporter

"""
#=================================#
   2D Data-Reco GD Optimization 
#=================================#
What this script does:
    (0) Parameter setting
    (1) Get A-Scans through forming H
        vec{A} = H^{T} \cdot b
            A = array with Nt x Nx, containing all A-Scans (i.e. B-Scan)
            H = array with L x L (L = Nt* Nx), collection of all SAFT matrix at each scan position
                H_i = i-th SAFT matrix with Nt x L, then
                H = [H_0^{T} H_1^{T} ..... H_{L-1}^{T}]
                -> all A-Scans for one possible defect position = row vector of H! 
            b = vec with the sieze L, vectorized defect map
    (2) Check H \cdot H^{T}?
    (3) Reco
"""
plt.close('all')
#======================================================================================================= Functions ===#

def display_time(stop):
    if stop > 60 and stop < 60**2:
        print('Time to complete : {} min'.format(round(stop/60, 2)))
    elif stop >= 60**2:
        print('Time to complete : {} h'.format(round(stop/(60**2), 2)))
    else:
        print('Time to complete : {} s'.format(round(stop, 2)))

        
def minimize_error_with_SGD(atrue, p_track, fwm_prog, analyzer, Niteration, err_max):
    """ Minimize the positional error using stochastic gradient descent(SGD) iteratively. SGD sloves the 
    least-squares problem to a linear matrix eqaution b = A \cdot x. In our case,
        b : collection of (atrue - amodel)
            difference b/w the measured A-Scan(atrue) and the modeled A-Scan based on the p_est(amodel)
        A : collection of curr_grad
            derivative of modeled A-Scans
        x : positional error [m], unitless
    Since the derivative of a modeled A-Scan is not linear, SGD works properly, only when the error is within
    the certain range which varies with the scan position (ptrue) and other measurement setups.
    
    Parameters
    ----------
        atrue : array-like (Nt)
            Measured A-Scan
        p_track : float, [m] unitless!
            Tracked scan position (p_track != p_true)
        fwm_prog : class
           DataGeneratorProgressive2D class, the first four steps should be done beforehand 
           (Cf. DataGeneratorProgressive2D)
        analyzer : class
            ImageQualityAnalyzerMSE class, initialization should be done with atrue beforehand
        Niteration : int
            Max number of iteration for SGD
    
    Returns
    -------
        curr_p_est : float, [m] unitless!
            Optimized scan position through iterative SGD
        aopt : array (Nt)
            1st Taylor approximation with the optimized scan position and the corresponding error
        se : float
            Normlized square error (Cf. ImageQualityAnalyzerMSE)
        
    """    
    curr_p_est = float(p_track) #[m], unitless
    print('#######################')
    print('Initial tracked position: {}mm'.format(curr_p_est* 10**3))
    #print('########')
    # The first column of A, b -> all 0, as it corresponds to err_x = 0
    A = np.zeros((atrue.shape[0], 1))
    b = np.zeros((atrue.shape[0], 1))
    
    # Set the initial se_min
    se_min = 1.0

    for n in range(Niteration):
        #print('Iteration No.{}'.format(n))
        fwm_prog.set_reflectivity_matrix(np.array([curr_p_est])*ureg.metre, oracle_cheating = True)
        Hmodel = fwm_prog.get_impulse_response_dictionary(without_refmatrix = True)
        amodel = fwm_prog.get_single_ascan(Hmodel)
        del Hmodel
        Hderiv = fwm_prog.get_dictionary_derivative(deriv_wrt_tau = False)
        curr_grad = -fwm_prog.get_single_ascan(Hderiv)
        del Hderiv
        
        # Update A and b
        A = np.append(A, np.array([curr_grad]).transpose(), axis = 1)
        b = np.append(b, np.array([atrue - amodel]).transpose(), axis = 1) 
        # Solve b = A \cdot x, only the first one is returned, as others (e.g. residual) are irrelevant
        err_xest = np.linalg.lstsq(A, b, rcond=None)[0] # rcond=None is given, becuase of the warning
        #print('Estimated error : {}mm'.format(round(err_xest[-1, -1]* 10**3, 3)))

        curr_aopt = amodel + curr_grad * err_xest[-1, -1]
        
        if abs(err_xest[-1, -1]) > err_max:
            curr_se = 1.0
        else:
            curr_se = analyzer.get_mse(np.array([curr_aopt]))
            curr_p_est = curr_p_est - err_xest[-1, -1]
        
        #print('Optimized scan position : {}mm'.format(round(curr_p_est* 10**3, 3)))
        #print('Current SE : {}'.format(curr_se))
        
        if se_min >= curr_se:
            se_min = float(curr_se)
            aopt = np.array(curr_aopt)
            p_est = float(curr_p_est + err_xest[-1, -1]) # add the error to reverse the error-adjustment
            perr_est = err_xest[-1, -1]
        
        if se_min <= 0.01:
            break
        
        elif curr_p_est < 0:
            break
    
    print("Final error:", round(se_min, 5))
    print('Final scan position: {}mm'.format(round(p_est* 10**3, 3)))
    return p_est, perr_est



#================================================================================================ Parameter Setting ===#
### Specimen params ###
Nxdata = 20
Ntdata = 680 #880
t0 = 6* 10**-6* ureg.second

c0 = 6300* ureg.metre/ ureg.second #6300
fS = 80* ureg.megahertz #80
openingangle = 25 # curretly does not do anything!

dx = 0.5 # [mm]
dxdata = dx*10**-3* ureg.metre
dzdata = 0.5* c0/(fS.to_base_units())
pos_defect = np.array([[10*dxdata.magnitude, 571*dzdata.magnitude]])* ureg.metre 


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
alpha = 20* ureg.megahertz**2 # prameter value from Jan

# Pulse setting for direct Gabor pulse calculation without refmatrix
pulse_params = {
        'fCarrier' : fCarrier,
        'fS' : fS,
        'alpha' : alpha
        }

# Stepsize for the SAFT dictionary 
stepsize = dxdata  #with unit

# for B-Scan generation -> the size of the ROI is required
Ntdict = Ntdata - int(t0 / ((1/fS).to_base_units()))
# for Reco
L = Ntdict* Nxdata

# for GD
wavelength = c0/fCarrier.to_base_units() # with unit
err_max = err_max = 5* wavelength.magnitude
Niteration = 15

# for error
seed = 5
np.random.seed(seed)
perr = 2.52 # [dx] -> 2dx = 1mm, for lambda error, perr = 2.52 


# File setting
# Version to call the proper Atrue
version = '190815'
# Error -> str
if type(perr) is int:
    perr_str = '{}mm'.format(int(perr* dx))
elif perr == 2.52:
    perr_str = 'lambda'
else:
    perr_str = {}
# base of fname storing each SAFT matrix H_i
fname_base = 'npy_data_storage/Hopt/saft_matrix_PosNo{}.npy'


#================================================================================================ A-Scan Generation ===#
# Load Atrue -> for esimate/improve error
Atrue = np.load('npy_data_storage/BScan/BScan_auto_{}.npy'.format(version))
# Base of all A-Scans, i.e. B-Scan
Aopt = np.zeros((Ntdict, Nxdata))
Adelta = np.zeros((Ntdict, Nxdata))
p_scan = [] # store all scan positions

# Iteration over scan positions
for idx in range(Nxdata):  
    print('*** Iteration No. {} ***'.format(idx))
    start = time.time()
    # Select current a_true -> analyzer
    atrue = np.array(Atrue[:, idx])
    # Call ImageQualityAnalyzer for SE calculation
    analyzer = ImageQualityAnalyzerMSE(np.array([atrue]))
    
    # Set p_track
    curr_p = np.array([idx  + perr* np.random.normal()])*dxdata
    
    # Call FWM
    fwm_prog = DataGeneratorProgressive2D(specimen_params, stepsize)
    fwm_prog.set_pulse_parameter(pulse_params)
    fwm_prog.unit_handling_specimen_params()
    fwm_prog.vectorize_defect_map()
    # Estimate error
    p_opt, perr_est = minimize_error_with_SGD(atrue, curr_p.magnitude, fwm_prog, analyzer, Niteration, err_max)
    p_scan.append(round(p_opt* 10**3, 3))
    
    # Calculte SAFT matrix w/ p_est and perr_est
    Hblock = fwm_prog.optimize_SAFT_dictonary(p_opt = p_opt, perr_opt = perr_est, oracle_cheating = True,
                                              deriv_wrt_tau = False)
    # Get A-Scan
    Aopt[:, idx] = fwm_prog.get_single_ascan(Hblock)
    Adelta[:, idx] = fwm_prog.get_single_ascan(fwm_prog.refmatrix)
    # save Hblock -> fatser calculation
    fname = fname_base.format(idx)
    #np.save(fname, Hblock)
    del Hblock
    
    stop = time.time() - start
    display_time(stop)
    print('#######################')


plt.figure(1)    
plt.imshow(Aopt)


#=============================================================================================== Position handling ===#
# Save scan positions as tuple
p_scan = tuple(p_scan)
# Quantized scan position
p_qt = np.array(p_scan) 
p_qt = np.around(p_qt/ dx)* dx # unitless, [mm]
p_qt = tuple(p_qt)
# Choose valid entries for reconstruction positions
p_valid = []
posidx_valid = [] # for PosNo
for idx, entry in enumerate(p_qt):
    # Within the valid range?
    if (entry < -0.5) or (entry >= Nxdata* dx):
        pass
    # Repeated entry?
    elif entry in p_valid:
        pass
    else:
        p_valid.append(entry)
        posidx_valid.append(idx)
        
# Save p_reco as tuple
p_valid = tuple(p_valid)
# Sortted version of p_valid for assigning the H into the right position
p_reco = list(p_valid)
p_reco.sort()
# Put PosNo into the corresponding order to p_reco
PosNo_all = []
for element in p_reco:
    # Find the corresnponding index of the p_valid (before sort)
    pidx = p_valid.index(element)
    # Add the corresponding index of p_scan for assigning the Hblock
    PosNo_all.append(posidx_valid[pidx])


#================================================================================== Choose valid entries of A-Scans ===#
# Base of valid A-Scans
Avalid = np.zeros((Ntdict, Nxdata)) #H = [H_0^{T} H_1^{T} ..... H_{L-1}^{T}], H_i = Ntdict x L

# Select only valid A-Scans
for curr_p, curr_no in zip(p_reco, PosNo_all):
    idx = int(curr_p/dx)
    Avalid[:, idx] = np.array(Atrue[:, curr_no])

plt.figure(2)    
plt.imshow(Avalid)


#======================================================================================== Form complete SAFT matrix ===#
# Base of all SAFT matrix H
H = np.load('npy_data_storage/Hauto_{}/Hauto.npy'.format(version)) #H = [H_0^{T} H_1^{T} ..... H_{L-1}^{T}], H_i = Ntdict x L

start = time.time()
# Load data through iteration over PosNo_all # reco does not work well -> because of the missing blocks?
for curr_p, curr_no in zip(p_reco, PosNo_all):
    idx = int(curr_p/dx)
    # load the current SAFT matrix
    curr_H = np.load(fname_base.format(curr_no))
    # Set the start&end for the current ditionary block
    block_start = idx* curr_H.shape[0]
    block_end = (idx + 1)* curr_H.shape[0]
    print('Block start: {}'.format(block_start))
    # Put the current dict into SAFT matrix
    H[:, block_start:block_end] = np.array(curr_H.T)


# Reco 
vecA = np.reshape(Avalid, L, order = 'F') # !!! do not forget 'F' !!!
#b = np.dot(np.linalg.pinv(H.T), vecA) #-> pseudo-inverse does not work!!!
b = np.dot(H, vecA) #-> reco-approximation works, though!!
reco = b.reshape(Ntdict, Nxdata, order = 'F')
# save data
np.save('npy_data_storage/Hopt/Hopt_{}_{}.npy'.format(version, perr_str), H)
np.save('npy_data_storage/Reco/Reco_opt_{}_{}.npy'.format(version, perr_str), reco)


plt.figure(4)
plt.imshow(reco)

stop = time.time() - start
print('#======================#')
print('*** Reconsturction ***')
display_time(stop)


