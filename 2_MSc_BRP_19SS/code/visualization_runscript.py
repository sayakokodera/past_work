# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

"""
#==================================#
Visualization for A-Scan estimation
#==================================#
Parameter used for data generation:
posscan_set_ID = [10, 15, 20, 30] = ['8.8mm', '10.9mm', '12.1mm', '15.8mm']
etrue_range = np.array([10, 50, 100])* 0.1mm
"""
# cloase all plots
plt.close('all')

# Load data
ftime_sepo = '20190516_16h39m38s'
# SE: proper vs offset (at different scan positions)
sepo10 = np.load('npy_data/mse/Approx/se_ID{}_PropOffset_{}.npy'.format(10, ftime_sepo))
sepo15 = np.load('npy_data/mse/Approx/se_ID{}_PropOffset_{}.npy'.format(15, ftime_sepo))
sepo20 = np.load('npy_data/mse/Approx/se_ID{}_PropOffset_{}.npy'.format(20, ftime_sepo))
sepo30 = np.load('npy_data/mse/Approx/se_ID{}_PropOffset_{}.npy'.format(30, ftime_sepo))

# SE: proper vs correction + approximation (at different scan positions)
# Dictionary formation at p_opt 
# p_opt = p_true + err_est -> a_appeox = (H(p_opt) + H_deriv(p_opt)* err_est)* def_vec
# Data loaded here shows the SE b/w the true and approximated A-Scans
ftime_sepa = '20190521_16h38m02s'
sepa10 = np.load('npy_data/mse/Correction+Approx/se_ID{}_{}.npy'.format(10, ftime_sepa))
sepa15 = np.load('npy_data/mse/Correction+Approx/se_ID{}_{}.npy'.format(15, ftime_sepa))
sepa20 = np.load('npy_data/mse/Correction+Approx/se_ID{}_{}.npy'.format(20, ftime_sepa))
sepa30 = np.load('npy_data/mse/Correction+Approx/se_ID{}_{}.npy'.format(30, ftime_sepa))

xvalues = sepa10[:, 0]
# Plot
#================ ID = 10 ================#
# Performance: offset vs approximation
plt.figure(1)
plt.plot(xvalues, sepo10[:, 1])
plt.plot(xvalues, sepa10[:, 1])
plt.xlabel('Error [mm]')
plt.ylabel('Squared Error')
plt.legend(['Proper vs Offset', 'Proper vs Approx'])
plt.title('Performance analysis at scan position: 8.8mm')
plt.savefig('plots/SE_performance_ID10.png')

#================ ID = 15 ================#
# Performance: offset vs approximation
plt.figure(2)
plt.plot(xvalues, sepo15[:, 1])
plt.plot(xvalues, sepa15[:, 1])
plt.xlabel('Error [mm]')
plt.ylabel('Squared Error')
plt.legend(['Proper vs Offset', 'Proper vs Approx'])
plt.title('Performance analysis at scan position: 10.9mm')
plt.savefig('plots/SE_performance_ID15.png')

#================ ID = 20 ================#
# Performance: offset vs approximation
plt.figure(3)
plt.plot(xvalues, sepo20[:, 1])
plt.plot(xvalues, sepa20[:, 1])
plt.xlabel('Error [mm]')
plt.ylabel('Squared Error')
plt.legend(['Proper vs Offset', 'Proper vs Approx'])
plt.title('Performance analysis at scan position: 12.1mm')
plt.savefig('plots/SE_performance_ID20.png')

#================ ID = 30 ================#
# proper vs offset
plt.figure(4)
plt.plot(xvalues, sepo30[:, 1])
plt.plot(xvalues, sepa30[:, 1])
plt.xlabel('Error [mm]')
plt.ylabel('Squared Error')
plt.legend(['Proper vs Offset', 'Proper vs Approx'])
plt.title('Performance analysis at scan position: 15.8mm')
plt.savefig('plots/SE_performance_ID30.png')

r"""
#================ Position dependency ================#
plt.figure(3)
plt.plot(sepa10[:, 1])
#plt.plot(sepa15[:, 1])
#plt.plot(sepa20[:, 1])
plt.plot(sepa30[:, 1])
plt.legend(['8.8mm', '15.8mm'])#['8.8mm', '10.9mm', '12.1mm', '15.8mm']
plt.title('SE: Proper vs Approximated')
"""