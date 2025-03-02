#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 16:10:25 2021

@author: sayakokodera
"""

import numpy as np

def search_vector(vec, mat, axis):
    return np.flatnonzero(np.all(np.isclose(mat, vec), axis = axis))