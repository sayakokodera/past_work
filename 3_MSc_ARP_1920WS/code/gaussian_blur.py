# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 22:06:36 2020

@author: johan
"""

import numpy as np

def gaussian_fct(vec, sigma, mu):
    return 1/(sigma* np.sqrt(2* np.pi))* np.exp(- (vec - mu)**2 / (2* sigma**2))

def gaussian_blur(shape0, shape1, p_mid, sigma0, sigma1):
    r"""
    """
    # y-axis (direction of shape[0])
    w_y = gaussian_fct(np.arange(shape0), sigma0, p_mid[0])
    W_y = np.reshape(np.repeat(w_y, shape1), (shape1, shape0) ,'F').T   
    # x-axis (direction of shape[1])
    w_x = gaussian_fct(np.arange(shape1), sigma1, p_mid[1])
    W_x = np.reshape(np.repeat(w_x, shape0), (shape0, shape1) ,'F')
    return W_x* W_y