# -*- coding: utf-8 -*-

import numpy as np
import pickle

from tools.datetime_formatter import DateTimeFormatter

"""
Modify parameter values in ML parameter dictionaries
"""


def paramvalue_modifier(ds_type):
    # Load the parameter dictionary
    fdate = '210503_{}'.format(ds_type)
    with open('params/ml_data_params_{}.pickle'.format(fdate), 'rb') as handle:
        ml_param_all = pickle.load(handle)
        
    # Parameters to be changed: SS coverage for uniform batch sampling
    n_freq = 20
    n_ss_single = 15
    if ds_type == 'train':
        sscov_uniform = np.array([1, 0.95, 0.95, 0.95, 0.9, 0.9, 0.9, 0.5, 0.5, 0.4, 0.4, 0.4, 0.3, 0.3, 0.3])
    else:
        sscov_uniform = np.repeat(np.array([0.9, 0.5, 0.3]), 5) 
    
    for setNo in range(len(ml_param_all)):
        
        param = ml_param_all[str(setNo)]
        
        if param['ss_method'] == 'uniform':
            print('#===========================================#')
            print('setNo = {} / {}'.format(setNo, len(ml_param_all)))
            print('sscov, was = {}'.format(param['ss_cov']))
            covNo = int(setNo/n_freq) % (2* n_ss_single)
            # Update
            param.update({
                'ss_cov' : sscov_uniform[covNo]
                    })
            print('sscov, is = {}'.format(param['ss_cov']))
            
            
    # Save the ML parameter dictionary
    dtf = DateTimeFormatter()
    today = dtf.get_date_str()
    with open('params/ml_data_params_{}_{}.pickle'.format(today, ds_type), 'wb') as handle:
        pickle.dump(ml_param_all, handle, protocol = pickle.HIGHEST_PROTOCOL) 
        
    del ml_param_all
        
    
paramvalue_modifier('vali')
paramvalue_modifier('test')
paramvalue_modifier('train')
