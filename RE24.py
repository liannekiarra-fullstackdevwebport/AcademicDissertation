import numpy as np
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


class RE24(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var = 2, n_obj = 2, n_constr = 0 , xl = np.array([0.5,0.5]), xu = np.array([4,50]))

    def _evaluate(self, x, out, *args, **kwargs):
        f = np.zeros(2)
        g = np.zeros(4)

        x1 = x[0]
        x2 = x[1]

        # First original objective function
        f[0] = x1 + (120 * x2)

        E = 700000
        sigma_b_max = 700
        tau_max = 450
        delta_max = 1.5
        sigma_k = (E * x1 * x1) / 100
        sigma_b = 4500 / (x1 * x2)
        tau = 1800 / x2
        delta = (56.2 * 10000) / (E * x1 * x2 * x2)

        g[0] = 1 - (sigma_b / sigma_b_max)
        g[1] = 1 - (tau / tau_max)
        g[2] = 1 - (delta / delta_max)
        g[3] = 1 - (sigma_b / sigma_k)
        g = np.where(g < 0, -g, 0)
        f[1] = g[0] + g[1] + g[2] + g[3]

        out["F"]= f
        out["G"]= g

problem = RE24()
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