import numpy as np
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



class RE32(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var = 4, n_obj =3, n_constr= 0 , xl = np.array([0.125, 0.1,0.1,0.125]), xu = np.array([5,10,10,5]))

    def _evaluate(self, x, out, *args, **kwargs):
        f = np.zeros(3)
        g = np.zeros(4)

        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]

        P = 6000
        L = 14
        E = 30 * 1e6

        # // deltaMax = 0.25
        G = 12 * 1e6
        tauMax = 13600
        sigmaMax = 30000

        # First original objective function
        f[0] = (1.10471 * x1 * x1 * x2) + (0.04811 * x3 * x4) * (14.0 + x2)
        # Second original objective function
        f[1] = (4 * P * L * L * L) / (E * x4 * x3 * x3 * x3)

        # Constraint functions
        M = P * (L + (x2 / 2))
        tmpVar = ((x2 * x2) / 4.0) + np.power((x1 + x3) / 2.0, 2)
        R = np.sqrt(tmpVar)
        tmpVar = ((x2 * x2) / 12.0) + np.power((x1 + x3) / 2.0, 2)
        J = 2 * np.sqrt(2) * x1 * x2 * tmpVar

        tauDashDash = (M * R) / J
        tauDash = P / (np.sqrt(2) * x1 * x2)
        tmpVar = tauDash * tauDash + ((2 * tauDash * tauDashDash * x2) / (2 * R)) + (tauDashDash * tauDashDash)
        tau = np.sqrt(tmpVar)
        sigma = (6 * P * L) / (x4 * x3 * x3)
        tmpVar = 4.013 * E * np.sqrt((x3 * x3 * x4 * x4 * x4 * x4 * x4 * x4) / 36.0) / (L * L)
        tmpVar2 = (x3 / (2 * L)) * np.sqrt(E / (4 * G))
        PC = tmpVar * (1 - tmpVar2)

        g[0] = tauMax - tau
        g[1] = sigmaMax - sigma
        g[2] = x4 - x1
        g[3] = PC - P
        g = np.where(g < 0, -g, 0)
        f[2] = g[0] + g[1] + g[2] + g[3]

        out["F"] = f
        out["G"] = g

problem = RE32()
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



