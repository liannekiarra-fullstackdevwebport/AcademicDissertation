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
from pymoo.visualization.pcp import PCP

class RE31(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var =3, n_obj= 3, n_constr = 0 , xl = np.array([1.e-05,1.e-05, 1.e+00]), xu = np.array([100,100,3]))
    def _evaluate(self, x, out, *args, **kwargs):
        f = np.zeros(3)
        g = np.zeros(3)

        x1 = x[0]
        x2 = x[1]
        x3 = x[2]

        # First original objective function
        f[0] = x1 * np.sqrt(16.0 + (x3 * x3)) + x2 * np.sqrt(1.0 + x3 * x3)
        # Second original objective function
        f[1] = (20.0 * np.sqrt(16.0 + (x3 * x3))) / (x1 * x3)

        # Constraint functions
        g[0] = 0.1 - f[0]
        g[1] = 100000.0 - f[1]
        g[2] = 100000 - ((80.0 * np.sqrt(1.0 + x3 * x3)) / (x3 * x2))
        g = np.where(g < 0, -g, 0)
        f[2] = g[0] + g[1] + g[2]

        out["F"]= f
        out["G"]= g


problem = RE31()
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



PCP().add(F).show()

