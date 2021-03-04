import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt
from scipy.stats import rv_continuous
from qiskit.circuit import Parameter, ParameterVector
from qiskit import *
from sklearn.metrics.cluster import adjusted_mutual_info_score as mi

class Layer:
    def __init__(self,name,num):