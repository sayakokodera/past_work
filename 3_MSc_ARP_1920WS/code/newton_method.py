# -*- coding: utf-8 -*-
"""
Newton Method
"""

import numpy as np

def Newton_method(x, grad_fx, inv_hess_fx):
    grad = grad_fx(x)
    inv_hess = inv_hess_fx(x)
    return x - np.dot(inv_hess, grad)