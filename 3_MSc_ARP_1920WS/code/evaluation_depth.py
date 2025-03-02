# -*- coding: utf-8 -*-
"""
    BEC Evaluation: Effect of the ROI depth
"""
import numpy as np
import matplotlib.pyplot as plt

from image_quality_analyzer import ImageQualityAnalyzerAPI
from image_quality_analyzer import ImageQualityAnalyzerSE
from image_quality_analyzer import ImageQualityAnalyzerGCNR
from tools.npy_file_writer import save_data
from tools.datetime_formatter import DateTimeFormatter 
import tools.tex_1D_visualization as vis

plt.close('all')

# Parameters
wavelength = 1.26*10**-3 #[m]
dx = 0.5*10**-3 # [m]
dy = dx

# Variables
zdef_all = np.arange(15, 81, 5) 
err_all = list(['1lambda'])
Nrealization = 200
errNo_all = np.arange(Nrealization)

# Parmeters related to API
thresholdAPI = 0.5
iqaAPI = ImageQualityAnalyzerAPI(wavelength, thresholdAPI, dx, dy)

# Parmeters related to GCNR
thresholdGCNR = 0.5
N_hist = 50 # Higher N_hist = finer p.d.f, >50 _> insignificant change

# Base of results
res_single = np.zeros((Nrealization, len(zdef_all)))
results = {
        'se': {
                'track': np.copy(res_single),
                'opt': np.copy(res_single)
                },
        'api': {
                'true': np.zeros(len(zdef_all)),
                'track': np.copy(res_single),
                'opt': np.copy(res_single)
                },
        'gcnr': {
                'true': np.zeros(len(zdef_all)),
                'track': np.copy(res_single),
                'opt': np.copy(res_single)
                }
        }

    
def evaluate_data(fname):
    data = np.load(fname)
    # SE
    se = iqaSE.get_se(data)
    # API
    api = iqaAPI.get_api(data)
    # GCNR
    gcnr = iqaGCNR.get_gcnr(data)
    del data
    return se, api, gcnr

for idx, zdef in enumerate(zdef_all):
    path = 'npy_data/depth/{}mm'.format(zdef)
    # Reference: Reco_true
    ftrue = '{}/Reco_true.npy'.format(path)
    # For MSE
    Reco_true = np.load(ftrue)
    iqaSE = ImageQualityAnalyzerSE(Reco_true)
    # For API
    results['api']['true'][idx] = iqaAPI.get_api(Reco_true)
    # For GCNR
    iqaGCNR = ImageQualityAnalyzerGCNR(N_hist)
    iqaGCNR.set_target_area(Reco_true, thresholdGCNR)
    results['gcnr']['true'][idx] = iqaGCNR.get_gcnr(Reco_true)  
    
    del Reco_true
    
    # Functions applied along the axis of errNo_all
    def evaluate_Reco_track(errNo):
        num = errNo[0].astype(int)
        fname = '{}/1lambda/Reco_track/No.{}.npy'.format(path, num)
        se, api, gcnr = evaluate_data(fname) 
        return se, api, gcnr 
    
    def evaluate_Reco_opt(errNo):
        num = errNo[0].astype(int)
        fname = '{}/1lambda/Reco_opt/No.{}.npy'.format(path, num)
        se, api, gcnr = evaluate_data(fname) 
        return se, api, gcnr 
     
    results['se']['track'][:, idx], results['api']['track'][:, idx], results['gcnr']['track'][:, idx] = \
            np.apply_along_axis(evaluate_Reco_track, 0, errNo_all[np.newaxis, :])
    results['se']['opt'][:, idx], results['api']['opt'][:, idx], results['gcnr']['opt'][:, idx] = \
            np.apply_along_axis(evaluate_Reco_opt, 0, errNo_all[np.newaxis, :])
 
                   
# Plots
plt.figure(1)
plt.title('MSE')
plt.plot(zdef_all, np.mean(results['se']['track'], 0), label = 'track')
plt.plot(zdef_all, np.mean(results['se']['opt'], 0), label = 'opt') 
plt.xlabel('Depth [mm]')
plt.legend()

plt.figure(2)
plt.title('API')
plt.plot(zdef_all, np.mean(results['api']['track'], 0), label = 'track')
plt.plot(zdef_all, np.mean(results['api']['opt'], 0), label = 'opt')
plt.plot(zdef_all, results['api']['true'], label = 'reference')
plt.xlabel('Depth [mm]')
plt.ylabel('API')
plt.legend()

plt.figure(3)
plt.title('GCNR')
plt.plot(zdef_all, np.mean(results['gcnr']['track'], 0), label = 'track')
plt.plot(zdef_all, np.mean(results['gcnr']['opt'], 0), label = 'opt')
plt.plot(zdef_all, results['gcnr']['true'], label = 'reference')
plt.xlabel('Depth [mm]')
plt.ylabel('GCNR')
plt.legend()


# Save data as npy & tex
met_all = ['se', 'api', 'gcnr']
dtype_all = ['true', 'track', 'opt']
dtf = DateTimeFormatter()
today = dtf.get_date_str()
colors = ['fri_green', 'tui_blue', 'tui_orange']
linestyles = ['line width = 1pt']
x_y_reverse = False
for met in met_all:
    for idx, dtype in enumerate(dtype_all):
        if met == 'se' and dtype == 'true':
            pass
        else:
            # Format the data (axis 0 = depth, axis 1 = result)
            data = np.zeros((len(zdef_all), 2))
            data[:, 0] = np.copy(zdef_all)
            if dtype == 'true':
                data[:, 1] = results[met][dtype]
            else:
                data[:, 1] = np.mean(results[met][dtype], 0)
            # Settng file names
            fnpy = '{}_{}_{}.npy'.format(met, dtype, today)
            ftex = 'tex_files/{}_{}_depth.tex'.format(met, dtype)
            # Save data as npy
            save_data(data, 'npy_data', fnpy)
            # Export to tex
            vis.generate_coordinates_for_addplot(data[:, 0], ftex, x_y_reverse, [colors[idx]], [linestyles[0]], 
                                                 data[:, 1])

