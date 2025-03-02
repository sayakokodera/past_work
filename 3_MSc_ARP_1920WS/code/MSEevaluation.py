r"""
SE evaluation of the blind error correction
"""

import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

# Functions
def get_err_str(curr_err_max):
    if curr_err_max < 0.1:
        err_str = '00{}'.format(int(curr_err_max*10**3))
    elif curr_err_max < 1:
        err_str = '0{}'.format(int(curr_err_max*10**3))
    else:
        err_str = '{}'.format(int(curr_err_max*10**3))
    return err_str

def get_se(date, err_str, data_type, track):
    if track == True:
        data = np.load('npy_data/{}/uniform/{}_lambda/se_{}_track.npy'.format(date, err_str, data_type))
    else:
        data = np.load('npy_data/{}/uniform/{}_lambda/se_{}_opt.npy'.format(date, err_str, data_type))
    return np.mean(data)


# Load data
date = '200130'
err_norm = np.around(np.arange(0.2, 1.01, 0.025), 3)
err_norm = np.delete(err_norm, 20) # reco_track is missing in err = 0.70 lambda
sea_track = np.zeros(err_norm.shape[0])
sea_opt = np.zeros(sea_track.shape)
ser_track = np.zeros(sea_track.shape)
ser_opt = np.zeros(sea_track.shape)

for idx, err in enumerate(err_norm):
    err_str = get_err_str(err)
    sea_track[idx] = get_se(date, err_str, 'a', track = True)
    sea_opt[idx] = get_se(date, err_str, 'a', track = False)
    ser_track[idx] = get_se(date, err_str, 'reco', track = True)
    ser_opt[idx] = get_se(date, err_str, 'reco', track = False)

# Plot
plt.figure(1)
plt.plot(err_norm, sea_track, label = 'track')
plt.plot(err_norm, sea_opt, label = 'opt')
plt.title('SE in A-Scan')
plt.ylabel('SE')
plt.xlabel('Error/lambda')
plt.legend()

plt.figure(2)
plt.plot(err_norm, ser_track, label = 'track')
plt.plot(err_norm, ser_opt, label = 'opt')
plt.title('SE in reco')
plt.ylabel('SE')
plt.xlabel('Error/lambda')
plt.legend()

plt.show()

