######### run script for ImageQualityAnalizer : pos_scan #########

import numpy as np
import collections
import multiprocessing 
import time

import tools.json_file_writer as jsonwriter
from image_quality_analyzer import ImageQualityAnalyzerMSE
from image_quality_analyzer import ImageQualityAnalyzerGCNR

r"""
for the case pos_scan :
each data has following attributes as a basic info :
    key : the short description of the data-set (notes_varmain_value)
    xvalue (var_main value)
    fname    

"""
# set collections for easier access of the info 
BasicInfo = collections.namedtuple('BasicInfo',[
        'key',
        'sigma',
        'fname',
        ])

### setting the sigma range ###
## 0mm ... 1mm (sigma = 0.01 ... 199)
sigma_range1 = np.arange(0.01, 2, 0.01)

## 1mm ... 1.5mm (sigma = 200 ... 300)
sigma_range2 = np.arange(2, 3.01, 0.02)

## 1.5mm .... 2mm (sigma = 305 .... 400)
sigma_arr = list(np.around(np.arange(3.09, 4, 0.05), 2))
sigma_range3 = np.zeros([len(sigma_arr)+2])
sigma_range3[0] = 3.05
for idx, value in enumerate(sigma_arr):
    idx += 1
    sigma_range3[idx] = value
sigma_range3[-1] = 4.0
# sum of all sigmas
sigma_range = np.concatenate((sigma_range1, sigma_range2, sigma_range3), axis = 0)


notes_list = ['np_2', 'np_3', 'np_4', 'np_5']
Nsamples = 10
ese_posscan = {}
# path for the file names
fpath = 'H:/2018_Sayako_Kodera_BA_Daten/npy_data/ESE/pos_scan/dim_4_posdef_0/dr'

# setting a dictionary for better access & collection of the data 
# for the case : 2... 10mm
# =============================================================================
# for curr_note in notes_list:
#     ese_posscan.update({
#                 curr_note : []
#                 })
#     for curr_sigma in sigma_range:           
#         curr_key = 'dr_{}_sigma_{}'.format(curr_note, curr_sigma)
#         if curr_sigma < 1  and curr_sigma > 0:
#             fname_varmain = 'sigma_0{}'.format(int(curr_sigma*10))
#         else : 
#             fname_varmain = 'sigma_{}'.format(int(curr_sigma*10)) 
#         curr_fname = 'npy_data/ESE/pos_scan/dim_4_posdef_0/dr/{}/cimg_max_{}.npy'.format(curr_note,fname_varmain)
#         
#         ese_posscan[curr_note].append({
#                         'basic_info' : BasicInfo(key = curr_key,
#                                                  value = curr_sigma,
#                                                  fname = curr_fname),
#                         'cimg' : None,
#                         'mse' : None
#                         })
# =============================================================================
                        
                        
# for the case where 2+ samples are available for each sigma (i.e. 0.1... 2mm (so far 15.11.18))
for curr_note in notes_list:
    ese_posscan.update({
                curr_note : []
                })
    for curr_sigma in sigma_range:  
        curr_key = 'dr_{}_sigma_{}'.format(curr_note, curr_sigma)
        if curr_sigma < 0.1  and curr_sigma > 0:
            fname_varmain = 'sigma_00{}'.format(int(curr_sigma*100))
        elif curr_sigma < 1  and curr_sigma >= 0.1:
            fname_varmain = 'sigma_0{}'.format(int(curr_sigma*100))
        else : 
            fname_varmain = 'sigma_{}'.format(int(curr_sigma*100)) 
        curr_fname = '{}/dx_{}/cimg_max_{}'.format(fpath, curr_note,fname_varmain)
        # curr_fnae : modification required!!!!! 
        
        ese_posscan[curr_note].append({
                        'basic_info' : BasicInfo(key = curr_key,
                                                 sigma = 0.5* curr_sigma,
                                                 fname = curr_fname),
                        'cimg' : None,
                        'mse' : None
                        })
    
    
# log the basic information
# =============================================================================
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
# jsonwriter.dict2json(log_dict, 'json_log_data/ese_posscan_20181113.json')
#     
# =============================================================================

# reference data : gridded
cimg_ref = np.load('npy_data/ESE/grid/cimg_max_05_oa_0.npy')

analyzer_type = 'GCNR' #'MSE'

start = time.time()

# MSE calculation
if analyzer_type == 'MSE':
    for key in ese_posscan :
        # call ImageQualityAnalyzer
        analyzer = ImageQualityAnalyzerMSE(cimg_ref)
        # base of the result (result[:, 0] = sigma, result[:, 1] = MSE)
        result = np.zeros((len(ese_posscan[key]), 2))
        
        # iterate over sigma
        for item_idx in range(len(ese_posscan[key])):
            curr_item = ese_posscan[key][item_idx]
            # base of current cimg
            mse_mean = 0.0
            fnamesimga = curr_item['basic_info'].fname
            # iterate over samples
            for sam in range(Nsamples):
                sam += 1 # sample = 0 is not available yet (21.11.18)
                # load current cimg
                curr_sample = np.load('{}_{}.npy'.format(fnamesimga, sam))
                curr_mse = analyzer.get_mse(curr_sample)
                mse_mean += 1/Nsamples* curr_mse
            # add xvalue (i.e. sigma) to the result
            result[item_idx, 0] = curr_item['basic_info'].sigma
            # add mses to the result
            result[item_idx, 1] = mse_mean
            # set fname for mses
            fname = '{}/analysis/mse_{}.npy'.format(fpath, key)
            np.save(fname, result)

# GCNR calculation
if analyzer_type == 'GCNR':
    # ROI setting (here within 3dB of teh peak in the cimg_ref)
    defect_map = np.array([[10, 9], [16, 18], [20, 23], [29, 32]])
    ref_analyzer = ImageQualityAnalyzerGCNR(cimg_ref)
    ref_analyzer.set_roi(cimg_ref, defect_map, 0.5)
    roi = ref_analyzer.roi
    roi=np.zeros(roi.shape)
    roi[8:13,7:12] = 1
    roi[14:19,16:21] = 1
    roi[18:23,21:26] = 1
    roi[27:32,30:35] = 1  
    
    for key in ese_posscan :
        # base of the result (result[:, 0] = sigma, result[:, 1] = GCNR)
        result = np.zeros((len(ese_posscan[key]), 2))
        
        # iterate over sigma
        for item_idx in range(len(ese_posscan[key])):
            curr_item = ese_posscan[key][item_idx]
            # base of current cimg
            gcnr_mean = 0.0
            fnamesimga = curr_item['basic_info'].fname
            # iterate over samples
            for sam in range(Nsamples):
                sam += 1 # sample = 0 is not available yet (21.11.18)
                # load current cimg
                curr_sample = np.load('{}_{}.npy'.format(fnamesimga, sam))
                # call ImageQualityAnalyzer
                analyzer = ImageQualityAnalyzerGCNR(curr_sample, roi)
                # get GCNR for the current image
                curr_gcnr = analyzer.get_gcnr()
                gcnr_mean += 1/Nsamples* curr_gcnr
            # add xvalue (i.e. sigma) to the result
            result[item_idx, 0] = curr_item['basic_info'].sigma 
            # add gcnrs to the result
            result[item_idx, 1] = gcnr_mean 
            # set fname for mses
            fname = '{}/analysis/gcnr_{}.npy'.format(fpath, key)
            np.save(fname, result)


      
        
#result = list(map(analyze_data, [element for element in ese_posscan]))

end = time.time()
comp_time = end - start
if comp_time > 60:
    print('Time to complete : {} min'.format(round(comp_time/60, 2)))
elif comp_time > 60**2:
    print('Time to complete : {} h'.format(round(comp_time/(60**2), 2)))
else:
    print('Time to complete : {} s'.format(round(comp_time, 2)))

# fname : 'H:/2018_Sayako_Kodera_BA_Daten/npy_data/ESE/pos_scan/dim_4_posdef_0/dr/np_5/cimg_max_sigma_120_1.npy'

