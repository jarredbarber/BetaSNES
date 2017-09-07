import numpy as np

def identity(z):
    return z

def linear(z):
    return np.cast(np.argsort(np.array(z)),np.float32)/float(len(z))
