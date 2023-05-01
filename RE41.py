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



class RE41():
    def __init__(self):
        self.problem_name = 'RE41'
        self.n_objectives = 4
        self.n_variables = 7
        self.n_constraints = 0
        self.n_original_constraints = 10

        self.lbound = np.zeros(self.n_variables)
        self.ubound = np.zeros(self.n_variables)
        self.lbound[0] = 0.5
        self.lbound[1] = 0.45
        self.lbound[2] = 0.5
        self.lbound[3] = 0.5
        self.lbound[4] = 0.875
        self.lbound[5] = 0.4
        self.lbound[6] = 0.4
        self.ubound[0] = 1.5
        self.ubound[1] = 1.35
        self.ubound[2] = 1.5
        self.ubound[3] = 1.5
        self.ubound[4] = 2.625
        self.ubound[5] = 1.2
        self.ubound[6] = 1.2

    def evaluate(self, x):
        f = np.zeros(self.n_objectives)
        g = np.zeros(self.n_original_constraints)

        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]
        x5 = x[4]
        x6 = x[5]
        x7 = x[6]

        # First original objective function
        f[0] = 1.98 + 4.9 * x1 + 6.67 * x2 + 6.98 * x3 + 4.01 * x4 + 1.78 * x5 + 0.00001 * x6 + 2.73 * x7
        # Second original objective function
        f[1] = 4.72 - 0.5 * x4 - 0.19 * x2 * x3
        # Third original objective function
        Vmbp = 10.58 - 0.674 * x1 * x2 - 0.67275 * x2
        Vfd = 16.45 - 0.489 * x3 * x7 - 0.843 * x5 * x6
        f[2] = 0.5 * (Vmbp + Vfd)

        # Constraint functions
        g[0] = 1 - (1.16 - 0.3717 * x2 * x4 - 0.0092928 * x3)
        g[1] = 0.32 - (0.261 - 0.0159 * x1 * x2 - 0.06486 * x1 - 0.019 * x2 * x7 + 0.0144 * x3 * x5 + 0.0154464 * x6)
        g[2] = 0.32 - (
                    0.214 + 0.00817 * x5 - 0.045195 * x1 - 0.0135168 * x1 + 0.03099 * x2 * x6 - 0.018 * x2 * x7 + 0.007176 * x3 + 0.023232 * x3 - 0.00364 * x5 * x6 - 0.018 * x2 * x2)
        g[3] = 0.32 - (0.74 - 0.61 * x2 - 0.031296 * x3 - 0.031872 * x7 + 0.227 * x2 * x2)
        g[4] = 32 - (28.98 + 3.818 * x3 - 4.2 * x1 * x2 + 1.27296 * x6 - 2.68065 * x7)
        g[5] = 32 - (33.86 + 2.95 * x3 - 5.057 * x1 * x2 - 3.795 * x2 - 3.4431 * x7 + 1.45728)
        g[6] = 32 - (46.36 - 9.9 * x2 - 4.4505 * x1)
        g[7] = 4 - f[1]
        g[8] = 9.9 - Vmbp
        g[9] = 15.7 - Vfd

        g = np.where(g < 0, -g, 0)
        f[3] = g[0] + g[1] + g[2] + g[3] + g[4] + g[5] + g[6] + g[7] + g[8] + g[9]

        return f
