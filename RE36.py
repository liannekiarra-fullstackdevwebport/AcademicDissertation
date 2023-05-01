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




class RE36():
    def __init__(self):
        self.problem_name = 'RE36'
        self.n_objectives = 3
        self.n_variables = 4
        self.n_constraints = 0
        self.n_original_constraints = 1

        self.lbound = np.full(self.n_variables, 12)
        self.ubound = np.full(self.n_variables, 60)

    def evaluate(self, x):
        f = np.zeros(self.n_objectives)
        g = np.zeros(self.n_original_constraints)

        # all the four variables must be inverger values
        x1 = np.round(x[0])
        x2 = np.round(x[1])
        x3 = np.round(x[2])
        x4 = np.round(x[3])

        # First original objective function
        f[0] = np.abs(6.931 - ((x3 / x1) * (x4 / x2)))
        # Second original objective function (the maximum value among the four variables)
        l = [x1, x2, x3, x4]
        f[1] = max(l)

        g[0] = 0.5 - (f[0] / 6.931)
        g = np.where(g < 0, -g, 0)
        f[2] = g[0]

        return f
