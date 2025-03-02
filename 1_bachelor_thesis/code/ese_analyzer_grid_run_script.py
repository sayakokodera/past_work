######### run script for ImageQualityAnalizer : grid #########

import numpy as np
import collections
import multiprocessing 
import time

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
        'oa',
        'fname',
        ])

oa_range = np.arange(5, 21)
ese_grid = {}
# path for the file names
fpath = 'H:/2018_Sayako_Kodera_BA_Daten/npy_data/ESE/grid/oa_5_20'


ese_grid.update({
            'oa' : []
            })
for curr_oa in oa_range:           
    curr_key = 'oa_{}'.format(curr_oa)
    curr_fname = '{}/cimg_max_05_{}.npy'.format(fpath, curr_key)
    
    ese_grid['oa'].append({
                    'basic_info' : BasicInfo(key = curr_key,
                                             oa = curr_oa,
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
for key in ese_grid :
    # reference data
    # grid? or error-free data?
    cimg_ref = np.load('npy_data/ESE/grid/cimg_max_05_oa_0.npy')
    # call ImageQualityAnalyzer
    analyzer = ImageQualityAnalyzer(cimg_ref)
    # base of the result (result[:, 0] = sigma, result[:, 1] = MSE)
    result = np.zeros((len(ese_grid[key]), 2))
    
    # iterate over sigma
    for item_idx in range(len(ese_grid[key])):
        curr_item = ese_grid[key][item_idx]
        # base of current cimg
        cimg_mean = np.zeros((cimg_ref.shape[0], cimg_ref.shape[1]))
        fnameoa = curr_item['basic_info'].fname
        # iterate over samples
        curr_sample = np.load(fnameoa)
        # add xvalue (i.e. sigma) to the result
        result[item_idx, 0] = curr_item['basic_info'].oa
        # add mses to the result
        result[item_idx, 1] = analyzer.get_mse(curr_sample) 
        # set fname for mses
        fname = '{}/analysis/grid_mse_{}.npy'.format(fpath, key)
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

