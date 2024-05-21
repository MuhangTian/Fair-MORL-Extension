"""
Store helper functions
"""
import numpy as np
import os
import argparse


class DiscreFunc:
    def __init__(self, alpha, growth_rate=1.001):
        self.alpha = alpha
        self.growth_rate = growth_rate  # Growth rate for the exponential increase in the interval size.

    def __call__(self, x):
        if not isinstance(x, np.ndarray):
            x = np.array([x])  # Convert scalar to numpy array if not already an array
        assert isinstance(x, np.ndarray), "x must be a numpy array"
        if self.growth_rate > 1: 
            scaled_x = x / self.alpha
            indices = np.round(np.log1p(scaled_x * (self.growth_rate - 1)) / np.log(self.growth_rate)).astype(int)
            # Calculate the accumulated discretization bins
            bins = self.alpha * (self.growth_rate ** indices - 1) / (self.growth_rate - 1)
            return bins
        else:
            return np.round(x / self.alpha) * self.alpha

class WelfareFunc:
    def __init__(self, welfare_func_name, nsw_lambda=None, p=None, threshold=5):
        self.welfare_func_name = welfare_func_name
        self.nsw_lambda = nsw_lambda
        self.p = p
        self.threshold = threshold
        self.check()
    
    def check(self):
        if self.welfare_func_name == "nash-welfare":
            assert self.nsw_lambda is not None, "nsw_lambda must be specified for nsw welfare function"
        elif self.welfare_func_name == "p-welfare":
            assert self.p is not None, "p must be specified for p-welfare function"
        elif self.welfare_func_name == "resource_damage_scalarization":
            assert self.threshold is not None, "threshold must be specified for resource_damage_scalarization function"
    
    def __call__(self, x):
        assert isinstance(x, np.ndarray), "x must be a numpy array"
        
        if self.welfare_func_name == "utilitarian":
            return np.sum(x)
        elif self.welfare_func_name == "egalitarian":
            return np.min(x)
        elif self.welfare_func_name == "nash-welfare":
            x = x + self.nsw_lambda
            x = np.where(x <= 0, self.nsw_lambda, x)  # replace any negative values or zeroes with lambda
            return np.sum(np.log(x))
        elif self.welfare_func_name == "p-welfare":
            return np.power(np.mean(x ** self.p), 1 / self.p)
        elif self.welfare_func_name == "resource_damage_scalarization":
            return self.resource_damage_scalarization(x)
        else:
            raise ValueError("Invalid welfare function name")
    
    def nash_welfare(self, x):
        assert isinstance(x, np.ndarray), "x must be a numpy array"
        return np.power(np.prod(x), 1 / len(x))

    def resource_damage_scalarization(self, x):
        # assert isinstance(x, np.ndarray), "x must be a numpy array"
        # assert len(x) == 2, "x must have length 2 (resources_collected, damage_taken)"
        # R = x[0]
        # D = x[1]
        # return R - max(0, (D - self.threshold) ** 3)
        assert isinstance(x, np.ndarray), "x must be a numpy array"
        assert len(x) == 2, "x must have length 2 (resources_collected, damage_taken)"
        R = x[0]
        D = x[1]
        beta = 1 - self.threshold
        return (R ** self.threshold) * ((1 / (D+1)) ** beta)

def is_file_on_disk(file_name):
    if not os.path.isfile(file_name):
        raise argparse.ArgumentTypeError("the file %s does not exist!" % file_name)
    else:
        return file_name

def is_positive_integer(value):
    parsed_value = int(value)
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return parsed_value

def is_positive_float(value):
    parsed_value = float(value)
    if parsed_value <= 0.0:
        raise argparse.ArgumentTypeError("%s must be a positive value" % value)
    return parsed_value

def is_within_zero_one_float(value):
    parsed_value = float(value)
    if parsed_value <= 0.0 or parsed_value >=1:
        raise argparse.ArgumentTypeError("%s must be within (0,1)" % value)
    return parsed_value

def is_file_not_on_disk(file_name):
    if os.path.isfile(file_name):
        raise argparse.ArgumentTypeError("the file %s already exists on disk" % file_name)
    else:
        return file_name

if __name__ == "__main__":
    alpha = 0.8
    func = DiscreFunc(alpha,1)
    test_values = np.array([2.4])
    
    result = func(test_values)
    print("Input Values:", test_values)
    print("Discretized Output:", result)

    # Example usage
    # x = np.array([2.3661, 1.38742049])  # Example reward vector [resources_collected, damage_taken]
    # threshold = 0.5  # Example threshold
    # wf = WelfareFunc(welfare_func_name="resource_damage_scalarization", threshold=threshold)
    # result = wf(x)
    # print(f"Scalarized reward: {result}")