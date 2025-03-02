# -*- coding: utf-8 -*-
"""
Saving data
"""
import numpy as np
import os

def save_data(data, path, fname):    
    if not os.path.exists(path):
        os.makedirs(path)
    
    np.save('{}/{}'.format(path, fname), data)
    print('Data saved!')
    

def num2string(number):
    if number < 10:
        numstr = '00{}'.format(number)
    elif number < 100:
        numstr = '0{}'.format(number)
    else:
        numstr = '{}'.format(number)
    return numstr



