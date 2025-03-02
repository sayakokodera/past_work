######### run script for ImageQualityAnalizer : z_scan #########

import numpy as np
import collections
import multiprocessing 
import time
import matplotlib.pyplot as plt

import tools.json_file_writer as jsonwriter
from image_quality_analyzer import ImageQualityAnalyzerMSE

r"""
for the case pos_scan 
each data has following attributes as a basic info :
    key : the short description of the data-set (notes_varmain_value)
    sigma (var_main value)
    fname    

"""

BasicInfo = collections.namedtuple('BasicInfo',[
        'key',
        'sigma',
        'fname',
        ])

dz = 10**3* 6300 /(2* 80* 10**6) # now in mm
    
    
# new!! 181209 with err_range = 1mm instead of dz 
#sigma_arr1 = np.arange(0.05, 0.101, 0.01) 
#sigma_arr2 = np.arange(0.2, 1.01, 0.1)
#sigma_range = np.concatenate((sigma_arr1, sigma_arr2), axis = None)
sigma_range = np.array([0.22, 0.25, 0.28])

notes_list = ['oa_0', 'oa_1', 'oa_2']
ese_zscan = {}
Nsamples = 10
# path for the file names
fpath = 'H:/2018_Sayako_Kodera_BA_Daten/npy_data/ESE/zscan/dim4_dm0_grid0'
#fpath = 'npy_data/ESE/zscan/dim4_dm0_grid0'

#
for curr_note in notes_list:
    ese_zscan.update({
                curr_note : []
                })
    for curr_sigma in sigma_range:           
        curr_key = '{}_sigma_{}'.format(curr_note, curr_sigma)
        if curr_sigma < 0.1  and curr_sigma > 0:
            fname_varmain = 'sigma_00{}'.format(int(curr_sigma*100))
        elif curr_sigma == 0.126:
            fname_varmain = 'sigma_0{}'.format(int(curr_sigma)*1000)
        elif curr_sigma < 1  and curr_sigma >= 0.1:
            fname_varmain = 'sigma_0{}'.format(int(curr_sigma*100))
        else : 
            fname_varmain = 'sigma_{}'.format(int(curr_sigma*100)) 
        curr_fname = '{}/{}/181221_2_3mm/cimg_max_{}'.format(fpath, curr_note,fname_varmain)
        
        ese_zscan[curr_note].append({
                        'basic_info' : BasicInfo(key = curr_key,
                                                 sigma = curr_sigma,
                                                 fname = curr_fname),
                        'cimg' : None,
                        'mse' : None
                        })
    
    
# =============================================================================
# # log the basic information
# log_dict = {}
# for key in ese_posscan :
#     log_dict.update({
#         key : {}
#     })
#     for idx in range(len(ese_posscan[key])):
#         log_dict[key].update({
#         str(idx) :  {
#             'key' : ese_posscan[key][idx]['basic_info'].key,
#             'value' : ese_posscan[key][idx]['basic_info'].value,
#             'fname' : ese_posscan[key][idx]['basic_info'].fname
#         }})  
# # save the log data into a json file
# jsonwriter.dict2json(log_dict, 'json_log_data/ese_zscan_oa80_20181115.json')
# =============================================================================
    



# functions for paralell processing
def analyze_data_mse(item):
    # load cimage
    curr_cimg = np.load(item['basic_info'].fname)
    # get mse between the curr_cimg and the cimg_ref
    curr_mse = analyzer.get_mse(curr_cimg)
    return curr_mse

start = time.time()

        
# MSE calculation
for key in ese_zscan :
    # reference data
    # grid? or error-free data?
    cimg_ref = np.load('npy_data/ESE/grid/cimg_max_05_oa_0.npy')
    # call ImageQualityAnalyzer
    analyzer = ImageQualityAnalyzerMSE(cimg_ref)
    # base of the result (result[:, 0] = sigma, result[:, 1] = MSE)
    result = np.zeros((len(ese_zscan[key]), 2))
    
    # iterate over sigma
    for item_idx in range(len(ese_zscan[key])):
        curr_item = ese_zscan[key][item_idx]
        # base of current cimg
        cimg_mean = np.zeros((cimg_ref.shape[0], cimg_ref.shape[1]))
        fnamesimga = curr_item['basic_info'].fname
        print(fnamesimga)
        # iterate over samples
        for sam in range(Nsamples):
            sam += 1 # sample = 0 is not available yet (21.11.18)
            curr_sample = np.load('{}_{}.npy'.format(fnamesimga, sam))
            cimg_mean += 1/Nsamples* curr_sample
        # add xvalue (i.e. sigma) to the result
        result[item_idx, 0] = curr_item['basic_info'].sigma
        # add mses to the result
        result[item_idx, 1] = analyzer.get_mse(cimg_mean) # modification required! see below!!!!
        # set fname for mses
        fname = '{}/analysis/mse_0203mm_{}.npy'.format(fpath, key)
        np.save(fname, result)

end = time.time()
comp_time = end - start
if comp_time > 60:
    print('Time to complete : {} min'.format(round(comp_time/60, 2)))
elif comp_time > 60**2:
    print('Time to complete : {} h'.format(round(comp_time/(60**2), 2)))
else:
    print('Time to complete : {} s'.format(round(comp_time, 2)))

curr_alpha = analyzer.alpha

r"""
What to be modified:
    - item['basic_info'].fname : should be until sigma value 
    s.a. 'H:/2018_Sayako_Kodera_BA_Daten/npy_data/ESE/pos_scan/dim_4_posdef_0/dr/{}/cimg_max_{}'.format(curr_note,fname_varmain)
    where fname_var = 'sigma_002' or so
    - curr_cimg = 0.1* sum(cimg0, cimg1, cimg2, cimg3, cimg4, cimg5, cimg6, cimg7, cimg8, cimg9, cimg10)
        how :
            cimg_mean = np.zeros((cimg_ref.shape.[0], cimg_ref.shape[1]))
            for sam in range(Nsample + 1):
                curr_sample = np.load('......_{}.npy'.foramt(sam))
                cimg_mean += curr_sample / Nsample
    - curr_mse = analyzer.get_mse(cimg_mean)
"""