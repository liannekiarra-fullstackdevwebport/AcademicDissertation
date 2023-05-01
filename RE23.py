import numpy as np
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


class re23():
    def __init__(self):
        self.problem_name = 'RE23'
        self.n_objectives = 2
        self.n_variables = 4
        self.n_constraints = 0
        self.n_original_constraints = 3

        self.ubound = np.zeros(self.n_variables)
        self.lbound = np.zeros(self.n_variables)
        self.lbound[0] = 1
        self.lbound[1] = 1
        self.lbound[2] = 10
        self.lbound[3] = 10
        self.ubound[0] = 100
        self.ubound[1] = 100
        self.ubound[2] = 200
        self.ubound[3] = 240

    def evaluate(self, x):
        f = np.zeros(self.n_objectives)
        g = np.zeros(self.n_original_constraints)

        x1 = 0.0625 * int(np.round(x[0]))
        x2 = 0.0625 * int(np.round(x[1]))
        x3 = x[2]
        x4 = x[3]

        # First original objective function
        f[0] = (0.6224 * x1 * x3 * x4) + (1.7781 * x2 * x3 * x3) + (3.1661 * x1 * x1 * x4) + (19.84 * x1 * x1 * x3)

        # Original constraint functions
        g[0] = x1 - (0.0193 * x3)
        g[1] = x2 - (0.00954 * x3)
        g[2] = (np.pi * x3 * x3 * x4) + ((4.0 / 3.0) * (np.pi * x3 * x3 * x3)) - 1296000
        g = np.where(g < 0, -g, 0)
        f[1] = g[0] + g[1] + g[2]

        return f

class RE23(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var=3, n_obj=2, n_constr = 0, xl = np.array([1,1,10,10]), xu = np.array([100,100,200,240]))

    def _evaluate(self, x , out, *args, **kwargs):
        f = np.zeros(self.n_objectives)
        g = np.zeros(self.n_original_constraints)

        x1 = 0.0625 * int(np.round(x[0]))
        x2 = 0.0625 * int(np.round(x[1]))
        x3 = x[2]
        x4 = x[3]

        # First original objective function
        f[0] = (0.6224 * x1 * x3 * x4) + (1.7781 * x2 * x3 * x3) + (3.1661 * x1 * x1 * x4) + (19.84 * x1 * x1 * x3)

        # Original constraint functions
        g[0] = x1 - (0.0193 * x3)
        g[1] = x2 - (0.00954 * x3)
        g[2] = (np.pi * x3 * x3 * x4) + ((4.0 / 3.0) * (np.pi * x3 * x3 * x3)) - 1296000
        g = np.where(g < 0, -g, 0)
        f[1] = g[0] + g[1] + g[2]

        out["F"] = f
        out["G"] = g


problem = RE23()
algorithm = NSGA2(
    pop_size=1200,
    n_offsprings=10,
    sampling=get_sampling("real_random"),
    crossover=get_crossover("real_sbx", prob=0.9, eta=15),
    mutation=get_mutation("real_pm", eta=20),
    eliminate_duplicates=True
)

termination = get_termination("n_gen", 100)
res = minimize(problem,
               algorithm,
               termination,
               seed=1,
               save_history=True,
               verbose=True)

X = res.X #this are the number of variables
F = res.F #this are the number of objectives

plot = Scatter()
plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)

plot.show()


