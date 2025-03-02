# -*- coding: utf-8 -*-
############ run script for error source evaluation (ESE) : Pressure ToF ############

import numpy as np
import matplotlib.pyplot as plt
import json
import time

import data_synthesizers_ese
from ultrasonic_imaging_python.reconstruction.saft_for_manual_scan import SAFTonGridforManualScan
from ultrasonic_imaging_python.definitions import units
ureg = units.ureg

"""
Workflow
========= data generatior =========
--------- parameter setting ---------
1. load json param_dict files in the main.py (params_const, params_variable, err_sources)
2. call the proper ErrorSourceEvaluator class
3. register constant parmaeters into the ErrorSourceEvaluator class
4. for-loop : sub-vars
5. register specimen variables into the ErrorSourceEvaluator class
6. for-loop : main-var
7. set file names
8. set var_main in the ESE class
9. call the forward model class (DataGeneratorManualScanZscanError) 
10. set defect positions
11. configure error
12. input parameters into the fwm class 
--------- get data ---------
13. generate data
========= reconstruction =========
14. call the SAFTEngineProgressiveUpdate3D / SAFTonGridforRandomScan class 
15. parameter setup for reco
(configure logger?)
16. get reco
17. get cimage
18. save the data (A-scans, reco, cimage in txt)
(19. log the param_dict (txt?))

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
# param_errs = json.loads(open(fname_errs).read())
sigma_arr = np.arange(0.01, 1.001, 0.01)
# new!! 181209 with err_range = 1mm instead of dz 
sigma_arr1 = np.arange(0.05, 0.101, 0.01) 
sigma_arr2 = np.arange(0.2, 1.01, 0.1)
sigma_arr = np.concatenate((sigma_arr1, sigma_arr2), axis = None)
# new! 181211
sigma_arr = np.arange(0.055, 0.2, 0.005) # change oa in json file!!!!!!! 


param_errs = {
        'sigma' : {'values' : sigma_arr}
        }

Nsamples = 10

# function for generating data
def iterate_over_vars_and_get_data(dataese_class, Nsamples):
    # iterate over var_sub
    for curr_var_sub in dataese_class.var_sub :
        # Setting up the number of iteration        
        Niteration = len(dataese_class.input_param_vars['specimen_variables'][curr_var_sub])
        # for the case, where the variable dict contains the info other than variable set
        if 'base_unit' in dataese_class.input_param_vars['specimen_variables'][curr_var_sub] :
            Niteration = Niteration - 1
        if 'base_grid' in dataese_class.input_param_vars['specimen_variables'][curr_var_sub] :
            Niteration = Niteration - 1
        if 'abbreviation' in dataese_class.input_param_vars['specimen_variables'][curr_var_sub] :
            Niteration = Niteration - 1
        
        # setup for file names
        abbrev_varsub = dataese_class.input_param_vars['specimen_variables'][curr_var_sub]['abbreviation']
        # check json file!!!!!
        idx = 0
        
        # (4)
        for var_sub_idx in range(Niteration) :
            # (5)
            dataese_class.var_set.update({
                            curr_var_sub : str(var_sub_idx)
                            })
            dataese_class.register_specimen_variables(dataese_class.input_param_vars)
                        
            # (6) iterate over var_main
            for curr_var_main in dataese_class.var_main['values']:
                # get multiple data for each sigma value
                for sample in range(Nsamples):
                    # get time
                    start_iter = time.time()
                    
                    sam = sample + 1
                    # (7) setup for file names   
                    if curr_var_main < 0.1  and curr_var_main > 0:
                        fname_varmain = 'sigma_00{}_{}'.format(int(curr_var_main*1000), sam)
                    elif curr_var_main < 0.2  and curr_var_main >= 0.1:
                        fname_varmain = 'sigma_0{}_{}'.format(int(curr_var_main*1000), sam)
                    elif curr_var_main < 1  and curr_var_main >= 0.2:
                        fname_varmain = 'sigma_0{}_{}'.format(int(curr_var_main*100), sam)
                    else : 
                        fname_varmain = 'sigma_{}_{}'.format(int(curr_var_main*100), sam) 
                       
                    fname_varsub = '{}_{}'.format(abbrev_varsub, idx)
                    fname_iter = '{}_{}'.format(fname_varmain, fname_varsub)
                    fname_data = 'npy_data/ESE/zscan/dim4_dm0_grid0/{}/data_{}.npy'.format(fname_varsub,fname_varmain) 
                    fname_data_small = 'npy_data/ESE/zscan/dim4_dm0_grid0/{}/data_{}.npy'.format(fname_varsub,
                                                                                                 fname_varmain)
                    fname_reco = 'npy_data/ESE/zscan/dim4_dm0_grid0/{}/reco_{}.npy'.format(fname_varsub,fname_varmain)
                    fpath = 'H:/2018_Sayako_Kodera_BA_Daten/npy_data/ESE/zscan/dim4_dm0_grid0'
                    # new! wih err-range = 11mm on 181209 
                    #fpath = 'npy_data/ESE/zscan/dim4_dm0_grid0'
                    fname_cimg_max = '{}/{}/181221_005_02mm/cimg_max_{}.npy'.format(fpath, fname_varsub, fname_varmain) 
                    print('current file name : "{}"'.format(fname_cimg_max))
                    
                    # (8) register var_main into the DataGeneratorESE class                                       
                    dataese_class.register_error_source_variables(curr_var_main)
                    # (9) call fwm class 
                    dataese_class.call_fwm_class()
                    # (11) configure error
                    dataese_class.fwm_class.error_configuration(err_range = dataese_class.err_range, initialize = False, 
                                                                with_unit = True, sigma = dataese_class.sigma, 
                                                                seed_value = dataese_class.seed_value)
                    # (12) set fwm parameters
                    dataese_class.input_parameters_into_fwm_class()
                    # (13) get data
                    data = dataese_class.fwm_class.calculate_bscan()
                    # save data                
                    #np.save(fname_data, data)   
                    
                    # get parameter dictionary
                    specimen_params = dataese_class.get_specimen_parameters()
                    
                    # reduce the size of the data
                    data_small = dataese_class.fwm_class.reduce_data_size(data, specimen_params['t_offset'])
                    #np.save(fname_data_small, data_small)
                    
                    zImage = -0.5* specimen_params['t_offset']* specimen_params['c0']
                    ############################################################################################ reco ######
                    # (14) call reco class
                    compute_saft = SAFTonGridforManualScan('3DSingleMedium', enable_file_IO = True)
                    # (15) parameter setup
                    reco_params = dict(specimen_params)
                    reco_params.update({
                        'dxreco' : specimen_params['dxdata'],
                        'dyreco' : specimen_params['dydata'],
                        'Nxreco' : specimen_params['Nxdata'],
                        'Nyreco' : specimen_params['Nydata'],
                        'Nzreco' : data_small.shape[0],
                        'Ntdata' : data_small.shape[0],
                        'zImage' : zImage
                        })                
                    compute_saft.set_forward_model_parameters(reco_params)
                    x_scan_idx = dataese_class.x_transducer_list
                    y_scan_idx = dataese_class.y_transducer_list
                    
                    # (16) get reco                    
                    reco = compute_saft.get_reco(data_small, x_scan_idx, y_scan_idx, save_data = False)
                    
                    # (17) cimage
                    cimg_max = compute_saft.generate_cscan_format(reco, summation = False, save_data = False)
                    # save reco & cimg                
                    #np.save(fname_reco, reco)               
                    #np.save(fname_cimg_sum, cimg_sum)
                    np.save(fname_cimg_max, cimg_max)
                    
                    # get time
                    end_iter = time.time()
                    print('time to complete a signle iteration : {}s'.format(round(end_iter - start_iter, 2)))
                    
                
                
            idx = idx + 1
    

############## z_scan ##############
#### (2) ####
default_var_set_1 =  {'dimension' : '4', 
                    'grid_reco' : '0',
                    'defect_map' : '0',
                    'Npoint' : '3',
                    'base_grid_size' : '0',
                    'sigma' : '0'} 
            
            
ese_zscan = data_synthesizers_ese.ErroredDataPressureZscan()
ese_zscan.input_parameter_dataset(param_const, param_vars, param_errs) 
var_main = ese_zscan.var_main
#### (3) ####
ese_zscan.register_constant_parameters(param_const)

#### (4) & (5) ####
# get time
start_all = time.time()
iterate_over_vars_and_get_data(ese_zscan, Nsamples)
# get time
end_all = time.time()

comp_time = end_all - start_all
if comp_time > 60 and comp_time < 60**2:
    print('Time to complete : {} min'.format(round(comp_time/60, 2)))
elif comp_time >= 60**2:
    print('Time to complete : {} h'.format(round(comp_time/(60**2), 2)))
else:
    print('Time to complete : {} s'.format(round(comp_time, 2)))


