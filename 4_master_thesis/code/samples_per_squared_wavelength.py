#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 17:06:43 2021

@author: sayakokodera
"""

import numpy as np

# Variable 
coverage = 1

# Measurement setting
c0 = 5900 #[m/S]
fC = 4.0*10**6 #[Hz] 
wavelength = c0/fC
dx = 0.5*10**-3 #[m], dy takes also the same value
N_batch = 10
Nx = 50

# Within a single batch 
samples_per_batch = int(coverage* N_batch**2)
N_batch_normed = N_batch* dx / wavelength
samples_per_sqwavelength_batch = samples_per_batch / (N_batch_normed**2)
print('Within a batch:')
print('{} samples / batch'.format(samples_per_batch))
print('{} samples / squared wavelength'.format(samples_per_sqwavelength_batch))

# Within the ROI 
samples_per_roi = int(coverage* Nx**2)
N_roi_normed = Nx* dx / wavelength
samples_per_sqwavelength_roi = samples_per_roi / (N_roi_normed**2)
print('Within the ROI:')
print('{} samples / ROI'.format(samples_per_roi))
print('{} samples / squared wavelength'.format(samples_per_sqwavelength_roi))



    