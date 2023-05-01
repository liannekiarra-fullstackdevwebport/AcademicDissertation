import numpy as np

class RE91():
    def __init__(self):
        self.problem_name = 'RE91'
        self.n_objectives = 9
        self.n_variables = 7
        self.n_constraints = 0
        self.n_original_constraints = 0

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
        # stochastic variables
        x8 = 0.006 * (np.random.normal(0, 1)) + 0.345
        x9 = 0.006 * (np.random.normal(0, 1)) + 0.192
        x10 = 10 * (np.random.normal(0, 1)) + 0.0
        x11 = 10 * (np.random.normal(0, 1)) + 0.0

        # First function
        f[0] = 1.98 + 4.9 * x1 + 6.67 * x2 + 6.98 * x3 + 4.01 * x4 + 1.75 * x5 + 0.00001 * x6 + 2.73 * x7
        # Second function
        f[1] = max(0.0, (1.16 - 0.3717 * x2 * x4 - 0.00931 * x2 * x10 - 0.484 * x3 * x9 + 0.01343 * x6 * x10) / 1.0)
        # Third function
        f[2] = max(0.0, (
                    0.261 - 0.0159 * x1 * x2 - 0.188 * x1 * x8 - 0.019 * x2 * x7 + 0.0144 * x3 * x5 + 0.87570001 * x5 * x10 + 0.08045 * x6 * x9 + 0.00139 * x8 * x11 + 0.00001575 * x10 * x11) / 0.32)
        # Fourth function
        f[3] = max(0.0, (
                    0.214 + 0.00817 * x5 - 0.131 * x1 * x8 - 0.0704 * x1 * x9 + 0.03099 * x2 * x6 - 0.018 * x2 * x7 + 0.0208 * x3 * x8 + 0.121 * x3 * x9 - 0.00364 * x5 * x6 + 0.0007715 * x5 * x10 - 0.0005354 * x6 * x10 + 0.00121 * x8 * x11 + 0.00184 * x9 * x10 - 0.018 * x2 * x2) / 0.32)
        # Fifth function
        f[4] = max(0.0, (
                    0.74 - 0.61 * x2 - 0.163 * x3 * x8 + 0.001232 * x3 * x10 - 0.166 * x7 * x9 + 0.227 * x2 * x2) / 0.32)
        # Sixth function
        tmp = ((
                           28.98 + 3.818 * x3 - 4.2 * x1 * x2 + 0.0207 * x5 * x10 + 6.63 * x6 * x9 - 7.77 * x7 * x8 + 0.32 * x9 * x10) + (
                           33.86 + 2.95 * x3 + 0.1792 * x10 - 5.057 * x1 * x2 - 11 * x2 * x8 - 0.0215 * x5 * x10 - 9.98 * x7 * x8 + 22 * x8 * x9) + (
                           46.36 - 9.9 * x2 - 12.9 * x1 * x8 + 0.1107 * x3 * x10)) / 3
        f[5] = max(0.0, tmp / 32)
        # Seventh function
        f[6] = max(0.0, (
                    4.72 - 0.5 * x4 - 0.19 * x2 * x3 - 0.0122 * x4 * x10 + 0.009325 * x6 * x10 + 0.000191 * x11 * x11) / 4.0)
        # EighthEighth function
        f[7] = max(0.0, (
                    10.58 - 0.674 * x1 * x2 - 1.95 * x2 * x8 + 0.02054 * x3 * x10 - 0.0198 * x4 * x10 + 0.028 * x6 * x10) / 9.9)
        # Ninth function
        f[8] = max(0.0, (
                    16.45 - 0.489 * x3 * x7 - 0.843 * x5 * x6 + 0.0432 * x9 * x10 - 0.0556 * x9 * x11 - 0.000786 * x11 * x11) / 15.7)

        return f
