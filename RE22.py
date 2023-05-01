
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



class RE22(ElementwiseProblem):
    feasible_vals = np.array(
        [0.20, 0.31, 0.40, 0.44, 0.60, 0.62, 0.79, 0.80, 0.88, 0.93, 1.0, 1.20, 1.24, 1.32, 1.40, 1.55, 1.58, 1.60,
         1.76, 1.80, 1.86, 2.0, 2.17, 2.20, 2.37, 2.40, 2.48, 2.60, 2.64, 2.79, 2.80, 3.0, 3.08, 3, 10, 3.16, 3.41,
         3.52, 3.60, 3.72, 3.95, 3.96, 4.0, 4.03, 4.20, 4.34, 4.40, 4.65, 4.74, 4.80, 4.84, 5.0, 5.28, 5.40, 5.53,
         5.72, 6.0, 6.16, 6.32, 6.60, 7.11, 7.20, 7.80, 7.90, 8.0, 8.40, 8.69, 9.0, 9.48, 10.27, 11.0, 11.06, 11.85,
         12.0, 13.0, 14.0, 15.0])

    def __init__(self):
        super().__init__(n_var = 3, n_obj =2, n_constr = 0 , xl = np.array([0.2,0,0]), xu = np.array([15,20,40]))


    def _evaluate(self, x ,out, *args, **kwargs):
        f = np.zeros(2)
        g = np.zeros(2)
        # Reference: getNearestValue_sample2.py (https://gist.github.com/icchi-h/1d0bb1c52ebfdd31f14b3e811328390a)
        idx = np.abs(np.asarray(self.feasible_vals) - x[0]).argmin()
        x1 = self.feasible_vals[idx]
        x2 = x[1]
        x3 = x[2]
        # First original objective function
        f[0] = (29.4 * x1) + (0.6 * x2 * x3)

        # Original constraint functions
        g[0] = (x1 * x3) - 7.735 * ((x1 * x1) / x2) - 180.0
        g[1] = 4.0 - (x3 / x2)
        g = np.where(g < 0, -g, 0)
        f[1] = g[0] + g[1]

        out["F"] = f
        out["G"] = g

problem = RE22()
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


