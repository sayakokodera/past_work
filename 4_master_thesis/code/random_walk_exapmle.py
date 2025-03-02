#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Random Walk example
    Goal = check the average coverage after batch random walk
"""
import numpy as np
from spatial_subsampling import batch_random_walk2D

N_batch = 10 # batch size
N_realizations = 1000 # number of realizations
N_steps_all = np.array([30, 50, 70, 90, 120, 160]) # variable = number of steps

# Stats
means = np.zeros(len(N_steps_all))
stds = np.zeros(len(N_steps_all))

for idx, N_steps in enumerate(N_steps_all):
    
    def random_walk_rate(counter):
        """ Calculate how many grid points withint the batch is covered by the obtained walk
        """
        walk = batch_random_walk2D(N_steps, N_batch)
        r = walk.shape[0] / (N_batch**2)
        return r
    
    rates = np.apply_along_axis(random_walk_rate, 0, np.array([np.arange(N_realizations)]))
    
    # Stats
    means[idx] = np.mean(rates)
    stds[idx] = np.std(rates)
    
print('Average = {}'.format(means))
print('sigma = {}'.format(stds))


