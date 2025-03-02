# -*- coding: utf-8 -*-
############ run script for error source evaluation (ESE) : Scan Ares (SA) ############

import numpy as np
import matplotlib.pyplot as plt
import json

import data_synthesizers_ese
import tools.txt_file_writer as fwriter
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
12. set missing parameters for reconstruction
13. get reco
14. get cimage
15. save the data (A-scans, reco, cimage in txt)
16. log the param_dict (txt?)

"""

# function for generating data
def iterate_over_vars_and_get_data(dataese_class):
    # iterate over var_sub
    idx = 0 # = iteration index for fname
    for curr_var_sub in dataese_class.var_sub:
        for var_sub_idx in range(len(dataese_class.input_param_vars['specimen_variables'][curr_var_sub])):
            dataese_class.var_set.update({
                            curr_var_sub : str(var_sub_idx)
                            })
            dataese_class.register_specimen_variables(dataese_class.input_param_vars)
            
            # (5) iterate over var_main
            for curr_var_main in dataese_class.var_main['values']:
                # register var_main into the err_ev class                                       
                dataese_class.register_error_source_variables(curr_var_main)
                # (6) set file names ---> coming soon
                ######## data generation ########
                # (7) call fwm class 
                dataese_class._call_fwm_class()
                dataese_class.set_defect_positions()
                # (8) set fwm parameters
                dataese_class.input_parameters_into_fwm_class()
                # (9) configure logger ---> coming soon
                # (10) get data
                data = dataese_class.fwm_class.get_data(save_data = False)
                # save data
                fname_data = 'test_errsev_sa_data_' + str(idx) + '.txt'
                fwriter.write_txt_file(data, 'Ntdata, Nxdata, Nydata', fname_data)
                # get specimen_parameters 
                specimen_params = dataese_class.get_specimen_parameters()
                ######## reconstruction #########
                # (11)
                compute_saft = SAFTonGridforManualScan('3DSingleMedium', enable_file_IO = False)
                # (12)
                reco_params = dict(specimen_params)
                reco_params.update({
                    'dxdata' : specimen_params['grid_size']*ureg.millimeter,
                    'dydata' : specimen_params['grid_size']*ureg.millimeter,
                    'dxreco' : specimen_params['grid_size']*ureg.millimeter,
                    'dyreco' : specimen_params['grid_size']*ureg.millimeter,
                    'Nxreco' : specimen_params['Nxdata'],
                    'Nyreco' : specimen_params['Nydata'],
                    'Nzreco' : specimen_params['Ntdata']
                    })
                compute_saft.set_reco_parameters(reco_params)
                x_scan = list(specimen_params['pos_scan'][0].magnitude)
                y_scan = list(specimen_params['pos_scan'][1].magnitude)
                # (13)
                reco = compute_saft.get_reco(data, x_scan, y_scan, save_data = False)
                # save reco
                fname_reco = 'test_errsev_sa_reco_' + str(idx) + '.txt'
                fwriter.write_txt_file(reco, 'Ntdata, Nxdata, Nydata', fname_reco)
                # iteration index for fname
                idx = idx + 1




############## scan area ##############
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
        'grid_size' : {'values' : [1,3], 'unit' : ureg.millimeter},
        'Npoint' : {'values' : [50, 100, 150]},
        'pos_sa' : {'values' : [20]}
        }                
#### (2) ####
ese_sa = data_synthesizers_ese.ErroredDataScanArea()
ese_sa.input_parameter_dataset(param_const, param_vars, param_errs) 
var_main = ese_sa.var_main
#### (3) ####
ese_sa.register_constant_parameters(param_const)
#### (4) --- (15) ####
iterate_over_vars_and_get_data(ese_sa)




