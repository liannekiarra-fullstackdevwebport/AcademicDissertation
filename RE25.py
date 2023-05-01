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

class RE25(ElementwiseProblem):
    feasible_vals = np.array(
        [0.009, 0.0095, 0.0104, 0.0118, 0.0128, 0.0132, 0.014, 0.015, 0.0162, 0.0173, 0.018, 0.02, 0.023, 0.025,
         0.028, 0.032, 0.035, 0.041, 0.047, 0.054, 0.063, 0.072, 0.08, 0.092, 0.105, 0.12, 0.135, 0.148, 0.162,
         0.177, 0.192, 0.207, 0.225, 0.244, 0.263, 0.283, 0.307, 0.331, 0.362, 0.394, 0.4375, 0.5])
    def __init__(self):
        super().__init__(n_var = 3, n_obj=2, n_constr = 0 ,xl =np.array([1,0.6,0.09]), xu = np.array([70,3,0.5]))

    def _evaluate(self, x, out, *args ,**kwargs):
        f = np.zeros(self.n_objectives)
        g = np.zeros(self.n_original_constraints)

        x1 = np.round(x[0])
        x2 = x[1]
        # Reference: getNearestValue_sample2.py (https://gist.github.com/icchi-h/1d0bb1c52ebfdd31f14b3e811328390a)
        idx = np.abs(np.asarray(self.feasible_vals) - x[2]).argmin()
        x3 = self.feasible_vals[idx]

        # first original objective function
        f[0] = (np.pi * np.pi * x2 * x3 * x3 * (x1 + 2)) / 4.0

        # constraint functions
        Cf = ((4.0 * (x2 / x3) - 1) / (4.0 * (x2 / x3) - 4)) + (0.615 * x3 / x2)
        Fmax = 1000.0
        S = 189000.0
        G = 11.5 * 1e+6
        K = (G * x3 * x3 * x3 * x3) / (8 * x1 * x2 * x2 * x2)
        lmax = 14.0
        lf = (Fmax / K) + 1.05 * (x1 + 2) * x3
        dmin = 0.2
        Dmax = 3
        Fp = 300.0
        sigmaP = Fp / K
        sigmaPM = 6
        sigmaW = 1.25

        g[0] = -((8 * Cf * Fmax * x2) / (np.pi * x3 * x3 * x3)) + S
        g[1] = -lf + lmax
        g[2] = -3 + (x2 / x3)
        g[3] = -sigmaP + sigmaPM
        g[4] = -sigmaP - ((Fmax - Fp) / K) - 1.05 * (x1 + 2) * x3 + lf
        g[5] = sigmaW - ((Fmax - Fp) / K)

        g = np.where(g < 0, -g, 0)
        f[1] = g[0] + g[1] + g[2] + g[3] + g[4] + g[5]

        out["F"]= f
        out["G"] = g

problem = RE25()
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





