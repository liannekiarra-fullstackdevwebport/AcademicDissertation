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

class RE61():
    def __init__(self):
        self.problem_name = 'RE61'
        self.n_objectives = 6
        self.n_variables = 3
        self.n_constraints = 0
        self.n_original_constraints = 7

        self.lbound = np.zeros(self.n_variables)
        self.ubound = np.zeros(self.n_variables)
        self.lbound[0] = 0.01
        self.lbound[1] = 0.01
        self.lbound[2] = 0.01
        self.ubound[0] = 0.45
        self.ubound[1] = 0.10
        self.ubound[2] = 0.10

    def evaluate(self, x):
        f = np.zeros(self.n_objectives)
        g = np.zeros(self.n_original_constraints)

        # First original objective function
        f[0] = 106780.37 * (x[1] + x[2]) + 61704.67
        # Second original objective function
        f[1] = 3000 * x[0]
        # Third original objective function
        f[2] = 305700 * 2289 * x[1] / np.power(0.06 * 2289, 0.65)
        # Fourth original objective function
        f[3] = 250 * 2289 * np.exp(-39.75 * x[1] + 9.9 * x[2] + 2.74)
        # Fifth original objective function
        f[4] = 25 * (1.39 / (x[0] * x[1]) + 4940 * x[2] - 80)

        # Constraint functions
        g[0] = 1 - (0.00139 / (x[0] * x[1]) + 4.94 * x[2] - 0.08)
        g[1] = 1 - (0.000306 / (x[0] * x[1]) + 1.082 * x[2] - 0.0986)
        g[2] = 50000 - (12.307 / (x[0] * x[1]) + 49408.24 * x[2] + 4051.02)
        g[3] = 16000 - (2.098 / (x[0] * x[1]) + 8046.33 * x[2] - 696.71)
        g[4] = 10000 - (2.138 / (x[0] * x[1]) + 7883.39 * x[2] - 705.04)
        g[5] = 2000 - (0.417 * x[0] * x[1] + 1721.26 * x[2] - 136.54)
        g[6] = 550 - (0.164 / (x[0] * x[1]) + 631.13 * x[2] - 54.48)

        g = np.where(g < 0, -g, 0)
        f[5] = g[0] + g[1] + g[2] + g[3] + g[4] + g[5] + g[6]

        return f