""""This code in for RE91 it due ot a high number of objectives, it cannot easily access and store the values
from the objective archive, resulting to a high computation time for each combination iteration"""

import numpy as np
from pymoo.core.problem import ElementwiseProblem
import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.factory import get_termination
from pymoo.optimize import minimize
import matplotlib.pyplot as plt
from numpy import array
from pymoo.visualization.pcp import PCP
from pymoo.factory import get_problem, get_reference_directions

from pymoo.visualization.pcp import PCP
from pymoo.visualization.scatter import Scatter
from pymoo.visualization.pcp import PCP
from pymoo.indicators.hv import Hypervolume
from sklearn.neural_network import MLPRegressor as mlp
from pymoo.util.running_metric import RunningMetric as RM

global nn1
global nn2
global nn3
global nn4
global nn5
global nn6
global nn7
global nn8
global nn9

global res_comb



class RE91(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var =7, n_obj = 9, n_constr= 0 , xl = np.array([0.5,0.45,0.5,0.5,0.875,0.4,0.4]), xu = np.array([1.5, 1.35, 1.5, 1.5, 2.625, 1.2, 1.2]))
    def _evaluate(self, x,out, *args, **kwargs):
        f = np.zeros(9)
        g = np.zeros(0)

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

        out["F"] = f
        out["G"] = g
class RE91RandomiseData(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var =7, n_obj = 9, n_constr= 0 , xl = np.array([0.5,0.45,0.5,0.5,0.875,0.4,0.4]), xu = np.array([1.5, 1.35, 1.5, 1.5, 2.625, 1.2, 1.2]))
    def _evaluate(self, x,out, *args, **kwargs):
        f = np.zeros(9)
        g = np.zeros(0)

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

        out["F"] = f
        out["G"] = g

def combine():
    global nn1
    global nn2
    global nn3
    global nn4
    global nn5
    global nn6
    global nn7
    global nn8
    global nn9

    testProblem = RE91RandomiseData()
    #get data from NSGA2
    algorithm = NSGA2(
        pop_size=1200,
        n_offsprings=10,
        sampling=get_sampling("real_random"),
        crossover=get_crossover("real_sbx", prob=0.9, eta=15),
        mutation=get_mutation("real_pm", eta=20),
        eliminate_duplicates=True
    )

    termination = get_termination("n_gen", 100)
    res = minimize(testProblem,
                   algorithm,
                   termination,
                   seed=1,
                   save_history=True,
                   verbose=True)

    variables = res.X

    objectives = res.F
    # split objectives
    x = len(objectives)

    obj1 = [i for i in range(x)]
    obj2 = [i for i in range(x)]
    obj3 = [i for i in range(x)]
    obj4 = [i for i in range(x)]
    obj5 = [i for i in range(x)]
    obj6 = [i for i in range(x)]
    obj7 = [i for i in range(x)]
    obj8 = [i for i in range(x)]
    obj9 = [i for i in range(x)]


    a = 0

    while a <= x-1:
        for i in objectives:
            temp = i
            test1 = temp[0]
            test1 = [test1]
            test2 = temp[1]
            test2 = [test2]
            test3 = temp[2]
            test4 = temp[3]
            test5 = temp[4]
            test6 = temp[5]
            test7 = temp[6]
            test8 = temp[7]
            test9 = temp[8]

            test3 = [test3]
            test4 = [test4]
            test5 = [test5]
            test6 = [test6]
            test7 = [test7]
            test8 = [test8]
            test9 = [test9]



            obj1[a]= test1
            obj2[a]= test2
            obj3[a] = test3
            obj4[a] = test4
            obj5[a] = test5
            obj6[a] = test6
            obj7[a] = test7
            obj8[a] = test8
            obj9[a] = test9


        a = a +1

    print("objective one")
    print(obj1)

    print("objective two")
    print(obj2)

    print("variables")
    print(variables)

    global nn1
    global nn2
    global nn3
    global nn4
    global nn5
    global nn6
    global nn7
    global nn8
    global nn9


    global objectiveArchive
    global moreF
    global res_comb


    nn1 = mlp(solver='lbfgs', alpha=1e-5,hidden_layer_sizes = (15,), random_state = 1)
    nn1.fit(obj1, variables)
    nn2 = mlp(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
    nn2.fit(obj2, variables)
    nn3 = mlp(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
    nn3.fit(obj3, variables)
    nn4 = mlp(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
    nn4.fit(obj4, variables)
    nn5 = mlp(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
    nn5.fit(obj5, variables)
    nn6 = mlp(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
    nn6.fit(obj6, variables)
    nn7 = mlp(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
    nn7.fit(obj7, variables)
    nn8 = mlp(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
    nn8.fit(obj8, variables)
    nn9 = mlp(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
    nn9.fit(obj9, variables)


    problem = RE91()
    res_comb = minimize(testProblem,
                   algorithm,
                   termination,
                   seed=1,
                   save_history=True,
                   verbose=True)
    moreX = res_comb.X
    moreF = res_comb.F
    print("initial var")
    print(variables)
    print("initial obj")
    print(objectives)
    print("more var")
    print(moreX)
    print("more obj")
    print(moreF)


    print("displaying more F", len(objectives))
    print(objectives)
    print("displaying more X", len(variables))
    print(variables)


    print("-----variables-----")
    print(variables)
    print(moreX)
    print("---concatenating---")
    variables = np.concatenate((variables, moreX), axis=0)
    print(variables)
    print(len(variables))
    objectives = np.concatenate((objectives, moreF), axis =0)
    print(objectives)
    print(len(objectives))

   #combine loop

    loops = 5
    b = 0

    #combination loop needs to modify an external variable that is used when running another iteration

    objectiveArchive = []
    variableArchive = []


    while b <= loops:
        #retrieved more values
        print("------------------------------ LOOP ONE -------------------------")
        res_comb = minimize(testProblem,
                            algorithm,
                            termination,
                            seed=1,
                            save_history=True,
                            verbose=True)
        moreX = res_comb.X
        #getting the original objective function values
        realObj = realObjectiveFunctionValues(moreX)

        moreF = res_comb.F
        #store values in archive

        objectives = np.concatenate((objectives, moreF), axis=0)
        objectives = np.concatenate((objectives, moreF ), axis=0)
        for i in objectives:
            temp = i
            objectiveArchive.append(i)

        print("------Objective Archive after nsga2-----", len(objectiveArchive))

        print(objectiveArchive)

        variables = np.concatenate((variables, moreX), axis =0)
        variables = np.concatenate((variables,moreX), axis = 0)

        for i in variables:
            temp = i
            variableArchive.append(i)

        #  acces objective archive outside loop
        print("------Objective Archive after nsga2-----", len(variableArchive))

        print(variableArchive)

        #  retrieve values again from both archives to train neural networks
        #  place the objective values from the archive to a local array

        tempObjArchive = []

        for i in objectiveArchive:
            temp = i
            tempObjArchive.append(i)

        tempVarArchive = []

        for i in variableArchive:
            temp = i
            tempVarArchive.append(i)

        tempVarArchive = array(tempVarArchive)
        tempObjArchive = array(tempObjArchive)

        print("Printing temp objective archive", len(tempObjArchive))
        print(tempObjArchive)
        print("printing temp variable archive", len(tempVarArchive))
        print(tempVarArchive)
        print("Length objective one", obj1)
        print("Length objective one", obj2)
        print("Length objective one", tempVarArchive)

        #  split values in objective values archive
        x = len(tempObjArchive)

        obj1 = [i for i in range(x)]
        obj2 = [i for i in range(x)]
        obj3 = [i for i in range(x)]
        obj4 = [i for i in range(x)]
        obj5 = [i for i in range(x)]
        obj6 = [i for i in range(x)]
        obj7 = [i for i in range(x)]
        obj8 = [i for i in range(x)]
        obj9 = [i for i in range(x)]


        a = 0

        testVariable =[]
        for i in tempVarArchive:
            temp = i
            testVariable.append(temp)

        testVariable = array(testVariable)

        while a <= x - 1:
            for i in tempObjArchive:
                temp = i
                test1 = temp[0]
                test1 = [test1]
                test2 = temp[1]
                test2 = [test2]
                obj1[a] = test1
                obj2[a] = test2

                test3 = temp[2]
                test3 = [test3]
                test4 = temp[3]
                test4 = [test4]
                test5 = temp[4]
                test5 = [test5]
                test6 = temp[5]
                test6 = [test6]
                test7 = temp[6]
                test7 = [test7]
                test8 = temp[7]
                test8 = [test8]
                test9 = temp[8]
                test9 = [test9]

                obj3[a] = test3
                obj4[a] = test4
                obj5[a] = test5
                obj6[a] = test6
                obj7[a] = test7
                obj8[a] = test8
                obj9[a] = test9
        a = a + 1


        nn1 = mlp(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
        nn1.fit(obj1, testVariable)
        nn2 = mlp(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
        nn2.fit(obj2, testVariable)
        nn3 = mlp(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
        nn3.fit(obj3, testVariable)
        nn4 = mlp(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
        nn4.fit(obj4, testVariable)
        nn5 = mlp(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
        nn5.fit(obj5, testVariable)
        nn6 = mlp(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
        nn6.fit(obj6, testVariable)
        nn7 = mlp(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
        nn7.fit(obj7, testVariable)
        nn8 = mlp(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
        nn8.fit(obj8, testVariable)
        nn9 = mlp(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
        nn9.fit(obj9, testVariable)


        print("objective function values at this point: ", objectives)
        print("length of objective function values during iteration: ", len(objectives))

        b = b +1

    objectiveArchive = array(objectiveArchive)
    variableArchive = array(variableArchive)

    print(variables)
    print(len(variables))
    print("--------All objective function values--------")
    print("Objective Total Values", len(objectives))
    print(objectives)

    print("----printing objective archive-----", len(objectiveArchive))
    print(objectiveArchive)
    print("----printing variable archive-----", len(variableArchive))
    print(variableArchive)

    FF = objectiveArchive
    finalVariableValues = variableArchive
    XX = finalVariableValues

    plt.figure(figsize=(10, 10))
    plt.scatter(FF[:, 0], FF[:, 1], s=30, facecolors='none', edgecolors='blue')
    plt.title("Objective Space RE21 (NSAG2 and Neural Network Combination)")
    plt.show()

    xl = np.array([1,1.4142135623730951,1.4142135623730951,1])
    xu = np.array([3,3,3,3])
    plt.figure(figsize=(10,10 ))
    plt.scatter(XX[:, 0], XX[:, 1], s=30, facecolors='none', edgecolors='r')
    plt.xlim(xl[0], xu[0])
    plt.ylim(xl[1], xu[1])
    plt.title("Design Space RE21 (Combination of NSGA2 and Neural Networks)")
    plt.show()

    running = RM(delta_gen = 10,
                 n_plots = 10,
                 only_if_n_plots = True,
                 key_press = False,
                 do_show= True
    )
    for algorithm in res_comb.history:
        running.notify(algorithm)

def originalObjectiveFunction(x):
    f = np.zeros(9)
    g = np.zeros(0)

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

def realObjectiveFunctionValues(variablevalues):
        """For a given set of variable values, the real objective values have to be retrieved from this function. This
         function will return an array of the original objective values for each variable set"""
        variableValues = variablevalues
        objectiveFunctionValues = [i for i in range (len(variablevalues))]

        # utilising length of the variable values data set
        length = len(variablevalues)

        x = 0

        while x <= length-1:
            temp = variableValues[x]
            objVal =originalObjectiveFunction(temp)
            objectiveFunctionValues[x] = objVal
            x = x+1

        objectiveFunctionValues = array(objectiveFunctionValues)
        objectiveFunctionValues = np.array(objectiveFunctionValues)

        return objectiveFunctionValues

combine()
