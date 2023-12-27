"""
Store helper functions
"""
import numpy as np


class DiscreFunc:
    def __init__(self, discre_alpha):
        self.discre_alpha = discre_alpha
    
    def __call__(self, x):
        assert isinstance(x, np.ndarray), "x must be a numpy array"
        return np.floor(x / self.discre_alpha) * self.discre_alpha


class WelfareFunc:
    def __init__(self, welfare_func_name, nsw_lambda=None, p=None):
        self.welfare_func_name = welfare_func_name
        self.nsw_lambda = nsw_lambda
        self.p = p
        self.check()
    
    def check(self):
        if self.welfare_func_name == "nsw":
            assert self.nsw_lambda is not None, "nsw_lambda must be specified for nsw welfare function"
        elif self.welfare_func_name == "p-welfare":
            assert self.p is not None, "p must be specified for p-welfare function"
    
    def __call__(self, x):
        assert isinstance(x, np.ndarray), "x must be a numpy array"
        
        if self.welfare_func_name == "utilitarian":
            return np.sum(x)
        elif self.welfare_func_name == "egalitarian":
            return np.min(x)
        elif self.welfare_func_name == "nsw":
            x = x + self.nsw_lambda
            x = np.where(x <= 0, self.nsw_lambda, x)  # replace any negative values or zeroes with lambda
            return np.sum(np.log(x))
        elif self.welfare_func_name == "p-welfare":
            return np.power(np.mean(x ** self.p), 1 / self.p)
        else:
            raise ValueError("Invalid welfare function name")