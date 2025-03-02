r"""
Convex Optimization HW: task2 visualization
"""
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

# task (1): SNR = 0, 5, ... 30dB
snr_all = np.arange(0, 31, 5)
Rmeth1_snr = np.zeros(snr_all.shape[0])
Rmeth2_snr = np.zeros(snr_all.shape[0])
Rzf_snr = np.zeros(snr_all.shape[0])
for idx, snr in enumerate(snr_all):
    data1 = np.load('npy_data/Rmeth1_{}dB_etaidx2.npy'.format(snr))
    data2 = np.load('npy_data/Rmeth2_{}dB_etaidx2.npy'.format(snr))
    datazf = np.load('npy_data/Rzf_{}dB.npy'.format(snr))
    Rmeth1_snr[idx] = np.mean(data1)
    Rmeth2_snr[idx] = np.mean(data2)
    Rzf_snr[idx] = np.mean(datazf)

plt.figure(1)
plt.stem(snr_all, Rmeth1_snr, 'r', markerfmt='rs', label = 'Method 1')
plt.stem(snr_all, Rmeth2_snr, 'b', markerfmt='bo', label = 'Method 2')
plt.stem(snr_all, Rzf_snr, 'g', markerfmt='gv', label = 'Zero-forcing')
plt.title('SNR vs Sum-rate')
plt.xlabel('SNR [dB]')
plt.ylabel('Mean sum-rate')
plt.legend()
plt.grid()
#plt.savefig('tex/plots/task2_SNR.eps', format = 'eps')


# task (2): eta = np.array([1, 0.1, 0.01, 10**-3, 10**-4])
eta_all = np.array([1, 0.1, 0.01, 10**-3, 10**-4])
Rmeth1_eta = np.zeros(eta_all.shape[0])
Rmeth2_eta = np.zeros(eta_all.shape[0])
for idx, eta in enumerate(eta_all):
    data1 = np.load('npy_data/Rmeth1_10dB_etaidx{}.npy'.format(idx))
    data2 = np.load('npy_data/Rmeth2_10dB_etaidx{}.npy'.format(idx))
    Rmeth1_eta[idx] = np.mean(data1)
    Rmeth2_eta[idx] = np.mean(data2)

print('Rmeth1_eta = {}'.format(Rmeth1_eta))
print('Rmeth2_eta = {}'.format(Rmeth2_eta))

plt.figure(2)
plt.stem(np.log10(eta_all), Rmeth1_eta[::-1], 'r', markerfmt='rs', label = 'Method 1') 
plt.stem(np.log10(eta_all), Rmeth2_eta[::-1], 'b', markerfmt='bo', label = 'Method 2') #markerfmt='go',
plt.stem(np.log10(eta_all), Rzf_snr[2]*np.ones(eta_all.shape[0]), 'g', markerfmt='gv', label = 'Zero-forcing') #markerfmt='go',
plt.title('Interference threshold (eta) vs Sum-rate')
plt.xlabel('log10(eta)')
plt.ylabel('Mean sum-rate')
plt.legend()
plt.ylim(ymin = 8.0)
plt.grid()
#plt.savefig('tex/plots/task2_eta.eps', format = 'eps')

plt.show()
