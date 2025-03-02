#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from image_quality_analyzer import ImageQualityAnalyzerAPI
from image_quality_analyzer import ImageQualityAnalyzerSE
from image_quality_analyzer import ImageQualityAnalyzerGCNR
from tools.npy_file_writer import save_data
from tools.datetime_formatter import DateTimeFormatter 
import tools.tex_1D_visualization as vis
import tools.array2image as arr2im


#%% Functions
def evaluate_data(fname, iqaSE, iqaGCNR):
    reco = load_reco(fname)
    cscan = convert2cscan(reco)
    # SE
    se = iqaSE.get_se(cscan)
    # GCNR
    gcnr = iqaGCNR.get_gcnr(cscan)
    del reco, cscan
    return se, gcnr


def load_reco(fname):
    reco = np.load(fname)
    return reco / np.abs(reco).max()

def convert2cscan(data_3d):
    cscan = np.max(np.abs(data_3d), axis = 0) 
    return cscan

def update_results(res_dict, item, result):
    res_dict.update({
            item : result
        })
    return res_dict
    

#%% Parameters related to analysis
thresholdGCNR = 0.5
N_hist = 20 # Higher N_hist = finer p.d.f, >50 -> insignificant change



#%% Ground truth
R_true = load_reco('npy_data/simulations/R_true.npy')#[:, 5:-5, :]
c_true = convert2cscan(R_true)

# Initialization
# MSE
iqaSE = ImageQualityAnalyzerSE(c_true)
# GCNR
iqaGCNR = ImageQualityAnalyzerGCNR(N_hist)
iqaGCNR.set_target_area(c_true, thresholdGCNR)

# IQA for R_true
gcnr_true = iqaGCNR.get_gcnr(c_true)


#%% 
coverage_all = np.array([10])#np.array([5, 6, 7, 8, 9, 10, 12, 14, 15, 17]) # in %
N_sets = 1
fdate = '210625'
dtypes = ['R_smp', 'R_idw', 'R_re_idw', 'R_fk', 'R_re_fk']
set_buffer = 19
# if set_buffer != 0:
#     raise ValueError('set_buffer is not 0!!')

# Base
res_se = {}
res_gcnr = {}

for item in dtypes:
    # Base
    mse = np.zeros(len(coverage_all))
    mgcnr = np.zeros(len(coverage_all))
    
    for idx, cov in enumerate(coverage_all):
        path = 'npy_data/simulations/210625_{}%'.format(int(cov))
        
        # Base
        se_all = np.zeros(N_sets)
        gcnr_all = np.zeros(N_sets)
        
        for setNo in range(N_sets):
            fname =  '{}/setNo_{}/{}.npy'.format(path, int(setNo + set_buffer), item)
            se_all[setNo], gcnr_all[setNo] = evaluate_data(fname, iqaSE, iqaGCNR)
            
        # Take the mean
        mse[idx] = np.mean(se_all)
        mgcnr[idx] = np.mean(gcnr_all)
        
    # Update the results
    res_se = update_results(res_se, item, mse)
    res_gcnr = update_results(res_gcnr, item, mgcnr)


#%% For a single set
for elem in res_gcnr:
    print('rGCNR: {} = {}'.format(elem, res_gcnr[elem]/gcnr_true))
    
for elem in res_se:    
    print('SE: {} = {}'.format(elem, res_se[elem]))

# Check the number of "ootimal" resmpling positions   
rth = 2 
rmse = np.load('{}/setNo_{}/rmse_FK.npy'.format(path, int(setNo + set_buffer)))
x, y = np.nonzero(rmse > rth* np.nanmean(rmse))
print('N_resmp_fk = {}'.format(len(x)))
# Check the number of NaNs
x_nan, y_nan = np.nonzero(np.isnan(rmse))
print('N_nans = {}'.format(len(x_nan)))


import sys
sys.exit()        
        
#%% Plots
# GCNR
plt.figure()
plt.plot(coverage_all, res_gcnr['R_smp']/gcnr_true, label = 'R_smp')
plt.plot(coverage_all, res_gcnr['R_idw']/gcnr_true, label = 'R_idw')
plt.plot(coverage_all, res_gcnr['R_re_idw']/gcnr_true, label = 'R_re_idw')
plt.plot(coverage_all, res_gcnr['R_fk']/gcnr_true, label = 'R_fk')        
plt.plot(coverage_all, res_gcnr['R_re_fk']/gcnr_true, label = 'R_re_fk')  
plt.legend()
plt.title('MGCNR relative to the ground truth')
plt.xlabel('Coverage in [%]')
plt.ylabel('rGCNR')      

# RMSE
plt.figure()
plt.plot(coverage_all, res_se['R_smp']/gcnr_true, label = 'R_smp')
plt.plot(coverage_all, res_se['R_idw']/gcnr_true, label = 'R_idw')
plt.plot(coverage_all, res_se['R_re_idw']/gcnr_true, label = 'R_re_idw')
plt.plot(coverage_all, res_se['R_fk']/gcnr_true, label = 'R_fk')        
plt.plot(coverage_all, res_se['R_re_fk']/gcnr_true, label = 'R_re_fk')  
plt.legend()
plt.title('RMSE compared to c_true')
plt.xlabel('Coverage in [%]')
plt.ylabel('RMSE')      
    
