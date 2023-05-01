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


class RE21(ElementwiseProblem):
    def __init__(selfself):
        super().__init__(n_var =4, n_obj = 2, n_constr = 0 ,xl = np.array([1,1.4142135,1.41421356,1]), xu = np.array([3,3,3,3]))

    def _evaluate(self, x, out, *Args, **kwargs):
        f = np.zeros(2)
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]

        F = 10.0
        sigma = 10.0
        E = 2.0 * 1e5
        L = 200.0

        f[0] = L * ((2 * x1) + np.sqrt(2.0) * x2 + np.sqrt(x3) + x4)
        f[1] = ((F * L) / E) * ((2.0 / x1) + (2.0 * np.sqrt(2.0) / x2) - (2.0 * np.sqrt(2.0) / x3) + (2.0 / x4))
        out["F"] = f


problem = RE21()
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





