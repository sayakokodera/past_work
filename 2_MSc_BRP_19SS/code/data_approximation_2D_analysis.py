# -*- coding: utf-8 -*-
import numpy as np
import time
import matplotlib.pyplot as plt

import tools.json_file_writer as jsonwriter
from tools.datetime_formatter import DateTimeFormatter
from ultrasonic_imaging_python.definitions import units
ureg = units.ureg
from ultrasonic_imaging_python.sako_tools.image_quality_analyzer import ImageQualityAnalyzerMSE

"""
#==================================#
   2D Data Approximation Analysis
#==================================#

With this script, the meausred(i.e. true) and the approximated A-Scan are compared.
For this purpose, squared error (SE) b/w these two is calculated.
The obtained SEs are stored in npy_data/mse/Correction+Approx as .npy data.

"""
# for file names
fdate = '20190521'
# Get the current time
dtformatter = DateTimeFormatter()
curr_time = dtformatter.get_time_str()

#======================================================================================================== Functions ===#

def get_squarederror(fname, curr_err, analyzer):
    r"""
    Parameters
    ----------
        fname : str
        curr_err : float
        analyzer : ImageQualityAnalyzerMSE class
    """
    ascan = np.load(fname)
    error = curr_err* 0.1 #[mm]
    se = analyzer.get_mse(np.array([ascan]))
    
    return np.array([error, se])


def analyze_approximation(data_ref, curr_posID, eest_range, fdate, curr_time, *args):
    r"""
    (1) Choose curr_posID 
    (2) Load a_true = data_ref[:, curr_posID]
    (3) Call the class analyzer = ImageQualityAnalyzerMSE(a_true)
    (4) Set the base of SEs
    (5) Put the se value for error free case 
        se_pos[0, 0] = 0
        se_pos[0, 1] = analyzer.get_mse(a_true)
    (6) Set the path name for loading each A-Scan 
        path =  'npy_data_storage/{}/ID_{}'.format(fdate, curr_posID)
    ### Iteration over eest_range ###
    (7) Put the se values into the corresponding se field 
        se_pos[curr_iteration+1] = get_squarederror(f_pos, curr_err, analyzer)
        se_neg[curr_iteration] = get_squarederror(f_neg, -curr_err, analyzer)
    ##################################
    (8) Flip the negative error
        se_neg = np.flip(se_neg, 0)
    (9) Combine positive and negative errors
        se = np.concatenate((se_neg, se_pos), axis = 0)
    (10) Save the se data
        fname = 'npy_data/mse/se_ID{}_{}.npy'.format(curr_posID, curr_time)
    """
    # Get a_ture
    a_true = data_ref[:, curr_posID]
    # Call the analyzer class
    analyzer = ImageQualityAnalyzerMSE(np.array([a_true]))
    # Base of squared errors
    se_pos = np.zeros((len(eest_range), 2)) # se for positive error, INCLUDE ErrFree
    se_neg = np.zeros((len(eest_range), 2)) # se for positive error, EXCLUDE ErrFree
    se_len = se_pos.shape[0] + se_neg.shape[0]
    se = np.zeros((se_len, 2))
    # Set the path for A-scan loading
    path = 'npy_data_storage/{}/ID_{}/Approx'.format(fdate, curr_posID)
    ### Iteration over err_est ###
    for eest_idx, err_est in enumerate(eest_range):
        # Set the currrent file names
        # Modify eest_idx, as the error range contains 0, i.e. eest_idx = 0 -> ErrFree
        # From 20190519 -> no modification required! 
        f_pos = '{}/positive/ascan_ErrNo_{}.npy'.format(path, eest_idx)
        f_neg = '{}/negative/ascan_ErrNo_{}.npy'.format(path, eest_idx)
        if err_est == 0:
            f_pos = '{}/positive/ascan_ErrNo_{}.npy'.format(path, 'ErrFree')
            f_neg = '{}/negative/ascan_ErrNo_{}.npy'.format(path, 'ErrFree')
        se_pos[eest_idx] = get_squarederror(f_pos, err_est, analyzer)
        se_neg[eest_idx] = get_squarederror(f_neg, -err_est, analyzer)
    
    # Remove the first (i.e. err_est = 0 = error free) row
    print('old se_neg[0] = {}'.format(se_neg[0]))
    se_neg = np.delete(se_neg, 0, 0)
    print('new se_neg[0] = {}'.format(se_neg[0]))
    # Flip the values of se_neg
    se_neg = np.flip(se_neg, 0)
    # Concatanate possitive and negative errors
    se = np.concatenate((se_neg, se_pos), axis = 0)   
    # Save the se data
    fname = 'npy_data/mse/Correction+Approx/se_ID{}_{}.npy'.format(curr_posID, curr_time)
    np.save(fname, se)
    print(se.shape[0])
    

#======================================================================================== Squared Error Calculation ===#
# Error b/w p_true and p_opt
eest_range = np.arange(0, 100.1, 0.2)

# Load ref data
data_ref = np.load('npy_data/data_2D_paramset_3031.npy')
# Choose the right scan position -> in the future, from the log json dictionary?
posscan_set_ID = [10, 15, 20, 30]
# SE calculation
# ID = 10
analyze_approximation(data_ref, posscan_set_ID[0], eest_range, fdate, curr_time)
# ID = 15
analyze_approximation(data_ref, posscan_set_ID[1], eest_range, fdate, curr_time)
# ID = 30
analyze_approximation(data_ref, posscan_set_ID[2], eest_range, fdate, curr_time)
# ID = 30
analyze_approximation(data_ref, posscan_set_ID[3], eest_range, fdate, curr_time)

