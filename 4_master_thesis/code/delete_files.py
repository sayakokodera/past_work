#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import shutil
import numpy as np
from tools.npy_file_writer import num2string
""" 
###### Scripct to delete multiple files & folders ######

Different methods
    * os.remove:
        used for removing a single file
    * os.rmdir:
        used for deleting an empty folder
        -> if the selected folder is not empty, error is raised
    * shutil.rmtree:
        used fo deleting a folder including its content
"""


#%% Functions
def get_dinfo_list(depth_all, Ndef_all, datasizes,  phi_all):
    dinfo_list = []
    for depth in depth_all:
        for idx, Ndef in enumerate(Ndef_all):
            for dataNo in range(datasizes[idx]):
                curr_path = 'depth_{}/Ndef_{}/{}'.format(depth, num2string(Ndef), num2string(dataNo))
                for phi in phi_all:
                    curr_folder = 'phi_{}'.format(num2string(phi))
                    dinfo_list.append('{}/{}'.format(curr_path, curr_folder))
                
    return dinfo_list



#%% Path settings
#path_base_output = 'npy_data/ML/training/output'
## Variables 
#depth_all = np.array([391, 647, 903]) # 1159, 135, 903, 391, 647
#Ndef_all = np.array([2, 5, 10]) # 2, 5, 10
#datasizes = np.array([200, 100, 100])# 200 for Ndef == 2, 100 for Ndef == 5, 10
#phi_all = np.array([30, 45, 60, 90, 120, 135, 150]) # [degree] 30, 45, 60, 90, 120, 135, 150, 180
#
## All folder paths
#dinfo_list = get_dinfo_list(depth_all, Ndef_all, datasizes, phi_all)
#
#
#for item in dinfo_list:
#    shutil.rmtree('{}/{}'.format(path_base_output, item))
#


import os

# Simulartion parameters
N_realization = 40
buffer_setNo = 0
coverage_all = 0.01* np.array([10, 15, 7, 5]) # in %
N_coverage = len(coverage_all)

# File setting
flist = ['p_resmp_krig', 'R_re_fk', 'rmse_FK', 'rmse_re_FK' ]
fdate = '210625'


def delete_files(setNo_vec):
    # Setup
    setNo = setNo_vec[0] % N_realization
    coverage = coverage_all[int(setNo_vec[0] / N_realization)]
    
    path = 'npy_data/simulations/{}_{}%/setNo_{}'.format(fdate, int(100*coverage), int(setNo + buffer_setNo))
    
    ### Remove ###
    for item in flist:
        os.remove('{}/{}.npy'.format(path, item))  
        

np.apply_along_axis(delete_files, 0, np.array([np.arange(N_coverage* N_realization)]))





