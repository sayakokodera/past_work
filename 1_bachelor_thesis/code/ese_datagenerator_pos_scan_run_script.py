# -*- coding: utf-8 -*-
############ run script for error source evaluation (ESE) : Scan Position Error ############

import numpy as np
import matplotlib.pyplot as plt
import json
import time
from datetime import date
import multiprocessing

import data_synthesizers_ese
import tools.json_file_writer as jsonfw
from ultrasonic_imaging_python.reconstruction.saft_for_manual_scan import SAFTonGridforManualScan
from ultrasonic_imaging_python.manual_scans.scan_positions_quantizer import quantize_scan_positions
from ultrasonic_imaging_python.definitions import units
ureg = units.ureg

"""
Workflow
============ parameter setting ============
1. load json param_dict files in the main.py (params_const, params_variable, err_sources)
2. call the appropriate ErrorSourceEvaluator
3. register paramters (except var_sub and var_main)
4. for-loop : sub-vars ('Npoint', 'dxreco(?)' as the var_main only effects the position manipulation
5. call the DataGeneratorManualScan class
6. set parameters 
7. define scan positions
============ data generation ============
9. generate data
10. for-loop : main-var (err_distance)
11. configure error
12. manipulate scan positions
============ reco ============
13. call the SAFTEngineProgressiveUpdate3D / SAFTonGridforRandomScan class 
14. get reco
15. get cimage
16. save the data (A-scans, reco, cimage in txt)
17. log the param_dict (txt?)

"""
plt.close('all')
#### (1) ####
# json data
fname_const = 'parameter_set/manual_scan_params_constant.json'
fname_vars = 'parameter_set/manual_scan_params_variables.json'
# fname_errs = ''

# load json, ds stands for data set
param_const = json.loads(open(fname_const).read())
param_vars = json.loads(open(fname_vars).read())

# set the sigma range
sigma_range1 = np.arange(0.05, 1, 0.05)
sigma_range2 = np.arange(1, 5, 0.1)
sigma_range3 = np.arange(5, 10, 1)
sigma_arr = np.concatenate((sigma_range1, sigma_range2, sigma_range3), axis = 0) # ---> check fname_coverage!

# for getting the coverage (181207)
sigma_arr = np.arange(0, 10.01, 0.01)
sigma_arr =np.array([0])
param_errs = {
        'sigma' : {'values' : sigma_arr}
        }

# define the number of samples for each sigma 
Nsamples = 10

# function for generating data
def iterate_over_vars_and_get_data(dataese_class, Nsamples):
    coverage = []
    log_dict =  {}
    
    default_var_set = dataese_class.var_set
    # iterate over var_sub
    for curr_var_sub in dataese_class.var_sub :
        # reset the var_set
        dataese_class.var_set = dict(default_var_set)
        
        # Setting up the number of iteration        
        Niteration = len(dataese_class.input_param_vars['specimen_variables'][curr_var_sub])
        # for the case, where the variable dict contains the info on the 'base_unit' or the 'base_grid'
        if 'base_unit' in dataese_class.input_param_vars['specimen_variables'][curr_var_sub] :
            Niteration = Niteration - 1
        if 'base_grid' in dataese_class.input_param_vars['specimen_variables'][curr_var_sub] :
            Niteration = Niteration - 1
        if 'abbreviation' in dataese_class.input_param_vars['specimen_variables'][curr_var_sub] :
            Niteration = Niteration - 1
            
        # setup for file names
        idx = 2 # check json file first!!!!
                
        for var_sub_idx in range(Niteration):
            
            # setup for file names
            varsub_str = dataese_class.input_param_vars['specimen_variables'][curr_var_sub]['abbreviation']
            fname_varsub = varsub_str + '_{}'.format(idx)
            fname_data = 'npy_data/ESE/pos_scan/dim_4_posdef_0/data_{}.npy'.format(fname_varsub)
            fname_data_small = 'npy_data/ESE/pos_scan/dim_4_posdef_0/data_small_{}.npy'.format(fname_varsub)
            fname_scan_map = 'npy_data/ESE/pos_scan/dim_4_posdef_0/scanmap_{}.npy'.format(fname_varsub)
            
            # add current var_sub value to the variable set
            dataese_class.var_set.update({
                            curr_var_sub : str(var_sub_idx)
                            })
            dataese_class.register_specimen_variables(dataese_class.input_param_vars)
            ############################################################################ fwm : data generation #########
            # call fwm class 
            dataese_class.call_fwm_class()       
            # (8) set fwm parameters
            dataese_class.seed_value = 0
            dataese_class.input_parameters_into_fwm_class()
            # get scan positions as a map
            scan_map = dataese_class.fwm_class.map_random_scan_positions()
            #np.save(fname_scan_map, scan_map)
            specimen_params = dataese_class.specimen_parameters
            
            # (10) get data
            #data = dataese_class.fwm_class.calculate_bscan()
            # reduce_size
            #data_small = dataese_class.fwm_class.reduce_data_size(data, specimen_params['t_offset'])
            data_small = np.load(fname_data_small)
            # adjust the spatial offset
            zImage = -0.5* specimen_params['t_offset']* specimen_params['c0']
            # save data            
            #np.save(fname_data, data)
            #np.save(fname_data_small, data_small)
            
            # for logging data 
            log_dict.update({
                    fname_varsub : {
                        'params_for_data_generation' : specimen_params
                        }
                    })
            
            # (5) iterate over var_main 
            for curr_var_main in dataese_class.var_main['values']:
                # setup for file names
                varmain_str = 'sigma'

                # get multiple data for each sigma value
                for sample in range(Nsamples):
                    # get time
                    start_iter = time.time()
                    
                    sam_idx = sample + 1
                    if curr_var_main < 0.1  and curr_var_main > 0:
                        fname_varmain = '{}_00{}_{}'.format(varmain_str, int(curr_var_main*100), sam_idx)
                    elif curr_var_main < 1  and curr_var_main >= 0.1:
                        fname_varmain = '{}_0{}_{}'.format(varmain_str, int(curr_var_main*100), sam_idx)
                    else : 
                        fname_varmain = '{}_{}_{}'.format(varmain_str, int(curr_var_main*100), sam_idx) 
                    # dr
                    fname_dr_reco = 'npy_data/ESE/pos_scan/dim_4_posdef_0/dr/{}/reco_{}.npy'.format(fname_varsub,
                                                                                                    fname_varmain)
                    fdir = 'H:/2018_Sayako_Kodera_BA_Daten/npy_data/ESE/pos_scan/dim_4_posdef_0/dr'
                    fname_dr_cimg_max = '{}/lambda_{}/cimg_max_{}.npy'.format(fdir, fname_varsub,fname_varmain)              
                    
                    print('current file name : "{}"'.format(fname_dr_cimg_max))
                    
                    # register var_main + other info for error configuration                                      
                    dataese_class.register_error_source_variables(curr_var_main)
                    # configure error    
                    if curr_var_main == 0:
                        dataese_class.err_range = 0* ureg.meter
                        dataese_class.sigma = 1 # just input, do nothing here as err_range = 0
                    dataese_class.fwm_class.error_configuration(err_range = dataese_class.err_range, initialize = False, 
                                                                with_unit = True, sigma = dataese_class.sigma, 
                                                                seed_value = dataese_class.seed_value)
                    # (10) manipulate the scan positions (obtained data = unitless)
                    dataese_class.fwm_class.manipulate_scan_positions()
                    x_scan_err = dataese_class.fwm_class.x_transducer_with_err # off the grid, unitless
                    y_scan_err = dataese_class.fwm_class.y_transducer_with_err # off the grid, unitless
                    
                    # get parameter dictionary
                    specimen_params = dataese_class.specimen_parameters
                    
                    
                    ######################################################################################### reco #########
                    # (11) call reco class
                    compute_saft = SAFTonGridforManualScan('3DSingleMedium', enable_file_IO = True)
                    # (12) adjust dxreco, dyreco to the curr_var_sub value, which is accesible as
                    # specimen_params['grid_reco']
                    reco_params = dict(specimen_params)
                    reco_params.update({
                        'dxreco' : specimen_params['grid_reco'],
                        'dyreco' : specimen_params['grid_reco'],
                        'Nxreco' : specimen_params['Nxdata'],
                        'Nyreco' : specimen_params['Nydata'],
                        'Nzreco' : data_small.shape[0],
                        'Ntdata' : data_small.shape[0],
                        'zImage' : zImage
                        })
                    # input parameters into SAFT class
                    compute_saft.set_forward_model_parameters(reco_params)
                    
                    # quantize the scan positions with curr_var_sub (specimen_params['grid_reco'])
                    x_scan_err_idx, y_scan_err_idx, idx_list = quantize_scan_positions(x_scan_err, y_scan_err, 
                                                                                       reco_params['dxreco'], 
                                                                                       reco_params['dyreco'], 
                                                                                       reco_params['Nxreco'],
                                                                                       reco_params['Nyreco'],
                                                                                       with_unit = True)
                    new_data = np.array(data_small)
                    new_data = np.delete(new_data, idx_list, axis = 1)
    
                    # (13) weight / discard the repeate entries (quantized scan positions)
                    # discard
                    new_data_dr, x_scan_err_idx_new, y_scan_err_idx_new = discard_repeated_entries(x_scan_err_idx, 
                                                                                                y_scan_err_idx, 
                                                                                                new_data)
                    
                    #reco_dr = compute_saft.get_reco(new_data_dr, x_scan_err_idx_new, y_scan_err_idx_new, 
                    #                                save_data = False)
                    #reco_dr = np.load(fname_dr_reco)
                    # cimg
                    #cimg_sum_dr = compute_saft.generate_cscan_format(reco_dr, summation = True, save_data = False)
                    #cimg_max_dr = compute_saft.generate_cscan_format(reco_dr, summation = False, save_data = False)
                    # save
                    #np.save(fname_dr_reco, reco_dr)                
                    #np.save(fname_dr_cimg_sum, cimg_sum_dr)
                    #np.save(fname_dr_cimg_max, cimg_max_dr)
                    
                    # weight
                    #new_data_w = average_repeated_entries_simple(x_scan_err_idx, y_scan_err_idx, new_data)
                    #reco_w = compute_saft.get_reco(new_data_w, x_scan_err_idx, y_scan_err_idx, save_data = False)                
                    # cimg
                    #cimg_sum_w = compute_saft.generate_cscan_format(reco_w, summation = True, save_data = False)
                    #cimg_max_w = compute_saft.generate_cscan_format(reco_w, summation = False, save_data = False)
                    #save
                    #np.save(fname_w_reco, reco_w)                
                    #np.save(fname_w_cimg_sum, cimg_sum_w)
                    #np.save(fname_w_cimg_max, cimg_max_w)
                    
                    # get time
                    end_iter = time.time()
                    print('time to complete a single iteration : {}s'.format(round(end_iter - start_iter, 2)))
                    coverage.append(len(x_scan_err_idx)/(reco_params['Nxreco']* reco_params['Nyreco']))
                
            idx = idx + 1
    # store the coverage for the given sigma --> change file name accordingly before run!!!!
    #fname_coverage = '{}/coverage/coverage_lambda_np2_5_20181206.npy'.format(fdir)
    fdir_win = 'npy_data/ESE/pos_scan/dim_4_posdef_0/dr/coverage'
    fname_coverage = '{}/coverage_sigma_0_181209.npy'.format(fdir_win)
    np.save(fname_coverage, np.array(coverage))
    


def find_repeated_entries(input_arr):
    seen =  []
    idx_list = []
    repeated_entries = {}
    for idx in range(len(input_arr)):
        entry = list(input_arr[idx])
        if entry in seen:
            idx_list.append(idx)
            repeated_entries[str(entry)].append(idx)
        else:
            seen.append(entry)
            entry_list = [idx]
            repeated_entries.update({
                str(entry) : entry_list
                })
    return seen, repeated_entries, idx_list


def discard_repeated_entries(x_scan_err_idx, y_scan_err_idx, data): 
    pos_scan_idx = [x_scan_err_idx, y_scan_err_idx]
    pos_scan_idx = np.array(pos_scan_idx).T
    new_pos_scan_idx, _, idx_list = find_repeated_entries(pos_scan_idx)
    new_pos_scan_idx = np.array(new_pos_scan_idx) # convert to array for slicing 
                
    x_scan_err_idx_new = new_pos_scan_idx[:, 0]
    y_scan_err_idx_new = new_pos_scan_idx[:, 1]
    
    # remove the same entries from the data as well
    new_data = np.array(data) 
    new_data = np.delete(new_data, idx_list, axis = 1)
    
    return new_data, x_scan_err_idx_new, y_scan_err_idx_new
     
 

def average_repeated_entries_simple(x_scan_err_idx, y_scan_err_idx, data):
    new_data = np.array(data)
    pos_scan_idx = [x_scan_err_idx, y_scan_err_idx]
    pos_scan_idx = np.array(pos_scan_idx).T
    
    seen, repeated_entries, _ = find_repeated_entries(pos_scan_idx)
    
    for entry in seen:
        Nrepeated = len(repeated_entries[str(entry)])
        for idx in repeated_entries[str(entry)]:
            new_data[:, idx] = new_data[:, idx] / Nrepeated

    return new_data
        

""" for multiprocessing ---> code should be modified so that the iterate_over... function does not contain the 
iteration over the var_main. This for-loop will be operated with p.map.
"""

############## scan positions ##############
#### (2) ####
default_var_set_3 =  {'dimension' : '4', 
                    'grid_reco' : '0',
                    'defect_map' : '0',
                    'Npoint' : '3',
                    'base_grid_size' : '0',
                    'sigma' : '0'}   
                                                       
ese_ps = data_synthesizers_ese.ErroredDataScanPositions(var_set = default_var_set_3)
ese_ps.input_parameter_dataset(param_const, param_vars, param_errs) 
var_main = ese_ps.var_main

#### (3) ####
ese_ps.register_constant_parameters(param_const)

# get time
start_all = time.time()
#### (4) & (5) ####
iterate_over_vars_and_get_data(ese_ps, Nsamples)
# get time
end_all = time.time()

# display the total computational time
comp_time = end_all - start_all
if comp_time > 60 and comp_time < 60**2:
    print('Time to complete : {} min'.format(round(comp_time/60, 2)))
elif comp_time >= 60**2:
    print('Time to complete : {} h'.format(round(comp_time/(60**2), 2)))
else:
    print('Time to complete : {} s'.format(round(comp_time, 2)))
    

