"""Utility Functions

Useful functions to use as activation functions for various layers as well 
as their derivatives for backward propagation.

All the functions are supplied in a dictionary for use in the main program

Required Modules: numpy
"""

import numpy as np

def relu(x: np.ndarray) -> np.ndarray:
    """
    Rectified Linear Unit
    """
    return max(0,x)

def drelu(x: np.ndarray) -> np.ndarray:
    """
    RELU Derivative
    """
    x[x<=0] = 0
    x[x>0] = 1
    return x
    
    
def leakyrelu(x: np.ndarray, c: float = 0.01) -> np.ndarray:
    """
    Leaky-RELU
    """
    return max(c*x, x)
    
def dleakyrelu(x: np.ndarray, c: float = 0.01) -> np.ndarray:
    """
    Leaky-RELU Derivative
    """
    x[x<=0] = 0.01
    x[x>0] = 1
    return x

def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Sigmoid
    """
    return 1/(1 + np.exp(-x))

def dsigmoid(x: np.ndarray) -> np.ndarray:
    """
    Sigmoid Derivative
    """
    return np.exp(-x)/np.power(1 + np.exp(-x), 2)

def linear(x: np.ndarray) -> np.ndarray:
    """
    Linear Function
    """
    return x

def dlinear(x: np.ndarray) -> np.ndarray:
    """
    Linear Derivative
    """
    return 1
    
util_funcs = {"relu": relu,
         "drelu": drelu,
         "lrelu": leakyrelu,
         "dlrelu": dleakyrelu,
         "sig": sigmoid,
         "dsig": dsigmoid,
         "linear": linear,
         "dlinear": dlinear}