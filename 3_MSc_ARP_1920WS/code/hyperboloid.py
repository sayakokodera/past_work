# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 20:22:43 2020

@author: perezeo
"""


import numpy as np
import numpy.linalg as nlin
import matplotlib.pyplot as plt

xres = 0.3e-3 #sampling grid resolution in m
c = 6300 #speed of sound in m/s
NT = 3*512 #number of samples in time axis
fs = 40e6 #sampling frequecy in Hz
T = 1/fs #sampling period in s
taxis = np.arange(NT)*T
zaxis = taxis*c/2

xs = 3.1*xres #scatterer x-axis location in m
zs = 80e-3 #scatterer depth in m

xvals = np.arange(-100,100,1)*xres #sampling positions
Nx = len(xvals) #number of sampling positions
xvals_noisy = xvals + np.random.normal(0,0.0008,Nx) #inaccurate x-axis locations

tvals = 2/c*np.sqrt(np.power(xvals-xs,2) + zs**2) #time delay per sampling position
tvals_noisy = tvals + np.random.normal(0,0.0000001,Nx) #inaccurate time delays
tdiscrete = np.round(tvals/T).astype(int)
Bscan = np.zeros((NT,Nx))
Bscan[tdiscrete,np.arange(Nx)] = 1
plt.plot(xvals*1e3, tvals*1e6)
plt.plot(xvals_noisy*1e3, tvals_noisy*1e6)
plt.xlabel('x axis distance [mm]')
plt.ylabel('time [$\mu$s]')


# %% Estimation section

#first, let's keep only a handful of measurement positions:
density = 1 #keep this percentage of sampling locations
k = int(Nx*density/100) #keep only this many sampling locations; need at least 2
subsample = np.random.choice(Nx, k, replace=False) #draw from a uniform distribution without replacement
tvals_noisy = tvals_noisy[subsample] #subsample and keep k locations
xvals_noisy = xvals_noisy[subsample] #subsample and keep k locations corresponding to the tvals we kept

#use TLS to find the parameters; exploit our knowledge on the model:
#c^2*tvals^2/4 - xvals^2 = -2xs*xvals + gamma;
#gamma = xs^2 + zs^2

#make sure we only use noisy data here!
c2t2_4 = c**2*np.power(tvals_noisy,2)/4 #size k
xvals2 = np.power(xvals_noisy,2) #size k
extended = np.zeros((k,3)) #size k x 3
extended[:,2] = c2t2_4-xvals2
extended[:,0] = xvals_noisy
extended[:,1] = 1

#this should be rank 2; the last of the singular vectors is colinear to the needed parameters
U,S,VH = nlin.svd(extended)
sol = -VH[2,0:2].conjugate()/VH[2,2]

xs_hat = sol[0]/-2
gamma = sol[1]
zs_hat = np.sqrt(gamma - xs_hat**2)

#apply forward model and compare; this time with the correct xvals (assuming we learned them somehow):
tvals_hat = tvals = 2/c*np.sqrt(np.power(xvals-xs_hat,2) + zs_hat**2)
decimation = 5 #for pretty plots
plt.plot(xvals[::decimation]*1e3, tvals_hat[::decimation]*1e6, 'g*')
plt.legend(('Clean','Noisy','From estimated scatterer'))
