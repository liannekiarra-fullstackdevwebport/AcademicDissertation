import numpy as np
from pymoo.core.problem import ElementwiseProblem
import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.factory import get_termination
from pymoo.optimize import minimize
import matplotlib.pyplot as plt
from pymoo.visualization.pcp import PCP
from pymoo.factory import get_problem, get_reference_directions
from pymoo.visualization.pcp import PCP
from pymoo.visualization.scatter import Scatter
from pymoo.visualization.pcp import PCP
from pymoo.indicators.hv import Hypervolume




class RE37():
    def __init__(self):
        self.problem_name = 'RE37'
        self.n_objectives = 3
        self.n_variables = 4
        self.n_constraints = 0
        self.n_original_constraints = 0

        self.lbound = np.full(self.n_variables, 0)
        self.ubound = np.full(self.n_variables, 1)

    def evaluate(self, x):
        f = np.zeros(self.n_objectives)

        xAlpha = x[0]
        xHA = x[1]
        xOA = x[2]
        xOPTT = x[3]

        # f1 (TF_max)
        f[0] = 0.692 + (0.477 * xAlpha) - (0.687 * xHA) - (0.080 * xOA) - (0.0650 * xOPTT) - (
                    0.167 * xAlpha * xAlpha) - (0.0129 * xHA * xAlpha) + (0.0796 * xHA * xHA) - (
                           0.0634 * xOA * xAlpha) - (0.0257 * xOA * xHA) + (0.0877 * xOA * xOA) - (
                           0.0521 * xOPTT * xAlpha) + (0.00156 * xOPTT * xHA) + (0.00198 * xOPTT * xOA) + (
                           0.0184 * xOPTT * xOPTT)
        # f2 (X_cc)
        f[1] = 0.153 - (0.322 * xAlpha) + (0.396 * xHA) + (0.424 * xOA) + (0.0226 * xOPTT) + (
                    0.175 * xAlpha * xAlpha) + (0.0185 * xHA * xAlpha) - (0.0701 * xHA * xHA) - (
                           0.251 * xOA * xAlpha) + (0.179 * xOA * xHA) + (0.0150 * xOA * xOA) + (
                           0.0134 * xOPTT * xAlpha) + (0.0296 * xOPTT * xHA) + (0.0752 * xOPTT * xOA) + (
                           0.0192 * xOPTT * xOPTT)
        # f3 (TT_max)
        f[2] = 0.370 - (0.205 * xAlpha) + (0.0307 * xHA) + (0.108 * xOA) + (1.019 * xOPTT) - (
                    0.135 * xAlpha * xAlpha) + (0.0141 * xHA * xAlpha) + (0.0998 * xHA * xHA) + (
                           0.208 * xOA * xAlpha) - (0.0301 * xOA * xHA) - (0.226 * xOA * xOA) + (
                           0.353 * xOPTT * xAlpha) - (0.0497 * xOPTT * xOA) - (0.423 * xOPTT * xOPTT) + (
                           0.202 * xHA * xAlpha * xAlpha) - (0.281 * xOA * xAlpha * xAlpha) - (
                           0.342 * xHA * xHA * xAlpha) - (0.245 * xHA * xHA * xOA) + (0.281 * xOA * xOA * xHA) - (
                           0.184 * xOPTT * xOPTT * xAlpha) - (0.281 * xHA * xAlpha * xOA)

        return f