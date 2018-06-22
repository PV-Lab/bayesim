import numpy as np
import pandas as pd
import math
from scipy.stats import norm
from itertools import product
from copy import deepcopy
import matplotlib as mpl
import matplotlib.pyplot as plt

class Pmf(object):
    """
    Class that stores a PMF capable of nested sampling / "adaptive mesh refinement".

    Stores probabilities in a DataFrame which associates regions of parameter space with probability values.
    """

    def __init__(self, params):
        """
        Args:
            params (:obj:`param_list`): param_list object containing parameters to be fit and associated metadata
        """

        
