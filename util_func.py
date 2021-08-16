# -*- coding: utf-8 -*-
"""Utility Functions

Useful functions to use as activation functions for various layers as well 
as their derivatives for backward propagation. Each returns an activation cache
containing Z for in the forward propagation step.

All the functions are supplied in a dictionary for use in the main program

Required Modules: numpy
"""

import numpy as np

def relu(x: np.ndarray) -> np.ndarray:
    """
    Rectified Linear Unit
    """
    return max(0,x), x

def drelu(x):
    """
    RELU Derivative
    """
    return max(0,1)
    
    
def leakyrelu(x: np.ndarray, c = 0.01) -> np.ndarray:
    """
    Leaky-RELU
    """
    return max(c*x, x), x
    
def dleakyrelu(x: np.ndarray, c = 0.01) -> np.ndarray:
    """
    Leaky-RELU Derivative
    """
    x[x<=0] = 0
    x[x>0] = 1
    return x

def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Sigmoid
    """
    return 1/(1 + np.exp(-x)),x

def dsigmoid(x: np.ndarray) -> np.ndarray:
    """
    Sigmoid Derivative
    """
    return np.exp(-x)/np.power(1 + np.exp(-x), 2)
