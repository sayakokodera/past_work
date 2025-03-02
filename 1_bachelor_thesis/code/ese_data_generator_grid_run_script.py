# -*- coding: utf-8 -*-
############ run script for error source evaluation (ESE) : Grid Size ############

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
1. load json param_dict files in the main.py (params_const, params_variable, err_sources)
2. call the appropriate ErrorSourceEvaluator
3. register paramters (except var_sub and var_main)
4. for-loop : sub-vars
5. for-loop : main-var
6. set file names
7. call the DataGeneratorForwardModel3D / DataGeneratorManualScan class
8. set parameters
9. configure logger
10. generate data
11. call the SAFTEngineProgressiveUpdate3D / SAFTonGridforRandomScan class 
12. parameter setup for reco
(configure logger?)
13. get reco
14. get cimage
15. save the data (A-scans, reco, cimage in txt)
16. log the param_dict (txt?)

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
param_errs = {
        'grid_size' : {'values' : [0.5], 'unit' : ureg.millimeter}
        }
store_value = [0.5, 1, 2, 4]
oa_range = np.arange(20, 21)

# function for generating data
def iterate_over_vars_and_get_data(dataese_class):
    fpath = 'H:/2018_Sayako_Kodera_BA_Daten/npy_data/ESE/grid/oa_5_20'

    # iterate over var_sub
    for curr_var_sub in dataese_class.var_sub :
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
        fname_varsub = dataese_class.input_param_vars['specimen_variables'][curr_var_sub]['abbreviation']
        
        for var_sub_idx in oa_range:
            dataese_class.var_set.update({
                            curr_var_sub : '0'#str(var_sub_idx)
                            })
            dataese_class.register_specimen_variables(dataese_class.input_param_vars)
            dataese_class.specimen_parameters.update({
                    'openingangle' : var_sub_idx
                    })
            
            # (5) iterate over var_main
            for curr_var_main in dataese_class.var_main['values']:
                # (6) setup for file names                
                if curr_var_main < 1 :
                    fname_varmain = '0{}'.format(int(10* curr_var_main))
                else :
                    fname_varmain = str(curr_var_main)
                fname_iter = "{}_{}_{}".format(fname_varmain, fname_varsub, var_sub_idx)
                fname_data = '{}/data_{}.npy'.format(fpath, fname_iter)
                fname_data_small = '{}/data_small_{}.npy'.format(fpath, fname_iter)
                fname_data_3D_small = '{}/data_small_3D_{}.npy'.format(fpath, fname_iter)
                fname_reco = '{}/reco_{}.npy'.format(fpath, fname_iter)
                fname_cimg_reco_sum = '{}/cimg_sum_{}.npy'.format(fpath, fname_iter)
                fname_cimg_reco_max = '{}/oatest/cimg_max_{}.npy'.format(fpath, fname_iter)
                fname_cimg_data_sum = '{}/data_cimg_sum_{}.npy'.format(fpath, fname_iter)
                fname_cimg_data_max = '{}/data_cimg_max_{}.npy'.format(fpath, fname_iter)
                
                # register var_main into the DataGeneratorESE class                                       
                dataese_class.register_error_source_variables(curr_var_main)
                # (7) call fwm class 
                dataese_class.call_fwm_class()
                # (8) set fwm parameters
                dataese_class.input_parameters_into_fwm_class()
                # (9) configure logger ---> coming soon
                # (10) get data
                data = dataese_class.fwm_class.calculate_bscan()
                #data = np.load(fname_data)
                # save data                
                #np.save(fname_data, data)
                
                # get parameter dictionary
                specimen_params = dataese_class.get_specimen_parameters()

                # reduce the size
                data_small = dataese_class.fwm_class.reduce_data_size(data, specimen_params['t_offset'])
                #np.save(fname_data_small, data_small)
                zImage = -0.5* specimen_params['t_offset']* specimen_params['c0']
                
                # get cimg for data
                #cimg_data_sum = dataese_class.get_cimg(data = data_small, summation = True)
                #cimg_data_max = dataese_class.get_cimg(data = data_small, summation = False)
                #np.save(fname_cimg_data_sum, cimg_data_sum)
                #np.save(fname_cimg_data_max, cimg_data_max)
                #data_3D = dataese_class.fwm_class.data_3D
                #np.save(fname_data_3D_small, data_3D)
                
                ############################################################################################# reco #####
                # (11) call reco class
                compute_saft = SAFTonGridforManualScan('3DSingleMedium', enable_file_IO = True)
                # (12) parameter setup
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
                x_scan = dataese_class.x_scan_idx
                y_scan = dataese_class.y_scan_idx
                # (13) get reco
                start = time.time()
                reco = compute_saft.get_reco(data_small, x_scan, y_scan, save_data = False)
                end = time.time()
                print('time to complete iteration : {}s'.format(round(end - start), 2))
                #np.save(fname_reco, reco)
                #reco = np.load(fname_reco)
                
                # cimage
# =============================================================================
#                 cimg_reco_sum = compute_saft.generate_cscan_format(reco, summation = True,
#                                                                    save_data = False)
# =============================================================================
                cimg_reco_max = compute_saft.generate_cscan_format(reco, summation = False,
                                                                   save_data = False)
                                
                # save reco & cimg                
                #np.save(fname_cimg_reco_sum, cimg_reco_sum)
                np.save(fname_cimg_reco_max, cimg_reco_max)
                print(fname_cimg_reco_max)
                
                


############## grid size ##############
#### (2) ####

                        
ese_grid = data_synthesizers_ese.ErroredDataGrid()
ese_grid.input_parameter_dataset(param_const, param_vars, param_errs) 
var_main = ese_grid.var_main
#### (3) ####
ese_grid.register_constant_parameters(param_const)


#### (4) & (5) ####
start_all = time.time()
iterate_over_vars_and_get_data(ese_grid)
end_all = time.time()

# display the total computational time
comp_time = end_all - start_all
if comp_time > 60 and comp_time < 60**2:
    print('Time to complete : {} min'.format(round(comp_time/60, 2)))
elif comp_time >= 60**2:
    print('Time to complete : {} h'.format(round(comp_time/(60**2), 2)))
else:
    print('Time to complete : {} s'.format(round(comp_time, 2)))


