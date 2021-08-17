# -*- coding: utf-8 -*-
"""NNsteps

The functional steps in producing a neural network i.e. parameter initialisation, forward
propagation, backward propagration.

All to be used in neural network model later.

Required Modules: numpy, util_func
"""

from typing import List, Callable

import numpy as np

import scripts.util_func as util_func #NOQA

def init_parameters(layer_dims: List[int]) -> dict:
    """
    Initiliase the weight and bias vectors into a dictionary

    Parameters
    ----------
    layer_dims : list
        node numbers in each layer, length of dimensions is the depth

    Returns
    -------
    dict
        the initalised parameters for each layer

    """
    parameters: dict = {}
        
    L: int = len(layer_dims) #number of layers
    
    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters["b" + str(l)] = np.zeros((layer_dims[l],1))
        
    return parameters

def lin_forward(A: np.ndarray, W: np.ndarray, b: np.ndarray) -> tuple:
    """
    Linear Forward Propagation step

    Parameters
    ----------
    A : np.ndarray
        activation of previous layer
    W : np.ndarray
        weights matrix
    b : np.ndarray
        bias vector

    Returns
    -------
    Z : np.ndarray
        Linear Activation
    cache : tuple
        cached values of (A, W, b, Z) for use in back propagation

    """
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache

def forward_prop(A_prev: np.ndarray,
                 W: np.ndarray,
                 b: np.ndarray,
                 activation: Callable = util_func.util_funcs['relu']) -> tuple:
    """
    Forward Propagation Step

    Parameters
    ----------
    A_prev : np.ndarray
        Activation of the Previous Layer
    W : np.ndarray
        Weights
    b : np.ndarray
        Bias vector
    activation : Callable, Optional
        activation funtion for layer, defaults to RELU

    Returns
    -------
    tuple
        Activation and cache
    """
    
    Z, linear_cache = lin_forward(A_prev, W, b)
    return activation(Z), (linear_cache, Z)



def L_model_forward(X: np.ndarray, parameters: dict, 
                    actfuncs: List[str]) -> tuple:
    """
    Forward Propagation for L-layer network

    Parameters
    ----------
    X : np.ndarray
        Input 
    parameters : dict
        W, b for every layer
    actfuncs : List[str]
        activation functions for use with funcs dictionairy in util_func

    Returns
    -------
    A : np.ndarray
        Activation of final layer
    caches : tuple
        cached values of A, W, b and Z.

    """
    caches = []
    L = len(parameters)//2
    A = X
    
    for l in range(1, L+1):
        A_prev = A
        W = parameters["W" + str(l)]
        b = parameters["b" + str(l)]
        A, cache = forward_prop(A_prev, W, b, activation = util_func.util_funcs[actfuncs[l]])
        caches.append(cache)

    return A, caches

#cost function
cost = lambda AL, Y, m: -1/m * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1-Y, np.log(1-AL)))

def compute_cost(AL: np.ndarray, Y: np.ndarray, func: Callable = cost) -> float:
    """
    Compute the cost function

    Parameters
    ----------
    AL : np.ndarray
        The output of the final layer of the neural network
    Y : np.ndarray
        The training example output
    func : function, optional
        The cost function to be used. The default is cost.

    Returns
    -------
    costval : float
        The calculated cost, which will then be optimised

    """
    m = Y.shape[1]
    costval = cost(AL, Y, m)
    costval = np.squeeze(costval)
    return costval
    
def lin_backward(dZ: np.ndarray, cache: tuple) -> tuple:
    """
    Linear Backward Propagation

    Parameters
    ----------
    dZ : np.ndarray
        Linear activation of layer
    cache : tuple
        cached values from forward propagation

    Returns
    -------
    tuple
        gradients of activation of following layer and those of the weights and bias

    """
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = 1/m * np.dot(dZ, A_prev.T)
    db = 1/m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T, dZ)
    
    return (dA_prev, dW, db)
    
def backward_prop(dA: np.ndarray, cache: tuple, actfunc: str) -> tuple:
    """
    Backward Propagation Module

    Parameters
    ----------
    dA : np.ndarray
        gradient of the activation
    cache : tuple
        cache containing A, W, b and Z
    actfunc : str
        the activation function for this layer whose derivative is then used

    Returns
    -------
    tuple
        derivatives of dA, dW, and db

    """
    linear_cache, activation_cache = cache
    dZ = np.multiply(dA, util_func.util_funcs['d' + actfunc](activation_cache))
    dA_prev, dW, db = lin_backward(dZ, linear_cache)
    
    return dA_prev, dW, db

#gradient of the cost function
dcost = lambda AL, Y: -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

def L_model_backward(AL: np.ndarray,
                     Y: np.ndarray,
                     caches: tuple,
                     actfuncs: List[str],
                     func: Callable = dcost) -> dict:
    """
    Backward Propagation

    Parameters
    ----------
    AL : np.ndarray
        output activation of layer L
    Y : np.ndarray
        output of training examples
    caches : tuple
        cached values of A, W, b and Z
    actfuncs : List[str]
        activation functions for each layer
    func : Callable, optional
        derivative of the cost function used. The default is dcost.

    Returns
    -------
    dict
        gradients of each weight and bias vector

    """
    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)
    
    dA_current = func(AL, Y)
    
    
    for l in reversed(range(L)):
        current_cache = caches[L]
        dA_prev_temp, dW_temp, db_temp = backward_prop(dA_current, current_cache, actfuncs[L])
        grads['A' + str(L-1)] = dA_prev_temp
        grads['W' + str(L)] = dW_temp
        grads['b' + str(L)] = db_temp
        dA_current = dA_prev_temp
        
    return grads
    
def update_parameters(params: dict, grads: dict, learning_rate: float = 0.01) -> dict:
    """
    Update Weights and biases using calculated gradients

    Parameters
    ----------
    params : dict
        parameters from previous iteration
    grads : dict
        gradients calculated from backward propagation
    learning_rate : float, optional
        The factor by which the gradients change the parameters in optimal direction. The default is 0.01.

    Returns
    -------
    dict
        updated parameters

    """
    parameters = params.copy()
    L = len(parameters)//2
    
    for l in range(L):
        parameters["W" + str(l)] = parameters["W" + str(l)] - grads["dW" + str(l)] * learning_rate
        parameters["b" + str(l)] = parameters["b" + str(l)] - grads["db" + str(l)] * learning_rate
        
    return parameters
        
        
