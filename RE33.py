

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



class RE33(ElementwiseProblem):

    def __init__(self):
        super().__init__(n_var = 4, n_obj= 3, n_constr=0, xl = np.array([55,75,1000,11]), xu = np.array([80,110,3000,20]))

    def _evaluate(self, x, out, *args, **kwargs):
        f = np.zeros(3)
        g = np.zeros(4)

        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]

        # First original objective function
        f[0] = 4.9 * 1e-5 * (x2 * x2 - x1 * x1) * (x4 - 1.0)
        # Second original objective function
        f[1] = ((9.82 * 1e6) * (x2 * x2 - x1 * x1)) / (x3 * x4 * (x2 * x2 * x2 - x1 * x1 * x1))

        # Reformulated objective functions
        g[0] = (x2 - x1) - 20.0
        g[1] = 0.4 - (x3 / (3.14 * (x2 * x2 - x1 * x1)))
        g[2] = 1.0 - (2.22 * 1e-3 * x3 * (x2 * x2 * x2 - x1 * x1 * x1)) / np.power((x2 * x2 - x1 * x1), 2)
        g[3] = (2.66 * 1e-2 * x3 * x4 * (x2 * x2 * x2 - x1 * x1 * x1)) / (x2 * x2 - x1 * x1) - 900.0
        g = np.where(g < 0, -g, 0)
        f[2] = g[0] + g[1] + g[2] + g[3]
        out["F"]= f
        out["G"]= g

problem = RE33()
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
plot.add(res.F, facecolor="none", edgecolor="green")
plot.show()





