import numpy as np

"""
This module defines several common functions for neural networks
"""

EPS = 1e-8


def hardlim(x: np.ndarray):
    return np.where(x >= 0, 1, 0)


def sigmoid(x: np.ndarray):
    return 1/(1+np.exp(-x))


def relu(x: np.ndarray):
    return np.maximum(x, np.zeros_like(x))


def leaky_relu(x: np.ndarray):
    return np.maximum(x, 0.1*x)


class Function:

    """
    This class binds a function and its several properties (e.g. gradient) into a single instance
    """

    def __call__(self, x):
        pass

    def grad(self, x):
        pass

    def inverse(self, x):
        pass

class Sigmoid(Function):

    def __call__(self, x):
        # The output value is clipped to prevent precision error
        return np.clip(sigmoid(x), EPS, 1-EPS)
    
    def grad(self, x):
        return self(x)*(1-self(x))
    
    def inverse(self, x):
        return np.log(x)-np.log(1-x)
    

class HardLim(Function):

    def __call__(self, x):
        return hardlim(x)
    
    def grad(self, x):
        return np.zeros_like(x)
    
    def inverse(self, x):
        # This function don't actually have an inverse, so don't call this
        raise RuntimeWarning("The HardLim function does not have an inverse. This method should not be called.")
    

class ReLU(Function):

    def __call__(self, x):
        return relu(x)
    
    def grad(self, x):
        # Technically this function is not differentiable at x=0
        # However, we simply ignore this issue by returning 0 when x=0
        return hardlim(x)
    
    def inverse(self, x):
        # This function don't have an inverse when x < 0, so be careful when calling this
        if x >= 0:
            return x
        

class LeakyReLU(Function):

    def __call__(self, x):
        return leaky_relu(x)
    
    def grad(self, x):
        # Technically this function is not differentiable at x=0
        # However, we simply ignore this issue by returning 0 when x=0
        return np.where(x >= 0, 1, 0.1)
    
    def inverse(self, x):
        if x >= 0:
            return x
        else:
            return 10*x
    

class Tanh(Function):

    def __call__(self, x):
        # The output value is clipped to prevent precision error
        return np.clip(np.tanh(x), -1+EPS, 1-EPS)
    
    def grad(self, x):
        return 1-np.power(self(x), 2)
    
    def inverse(self, x):
        return 0.5*(np.log(1+x)-np.log(1-x))

    
FUNCTIONS = {"ReLU": ReLU(), "Sigmoid": Sigmoid(), "Tanh": Tanh(), "Leaky ReLU": LeakyReLU()}