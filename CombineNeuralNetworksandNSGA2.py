#this code works
import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.factory import get_termination
from pymoo.optimize import minimize
import matplotlib.pyplot as plt
from numpy import array
from pymoo.factory import get_problem, get_reference_directions
from sklearn.neural_network import MLPRegressor as mlp
from pymoo.util.running_metric import RunningMetric as RM
from pymoo.visualization.scatter import Scatter


global nn1
global nn2
global res_comb

class MyProblem(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var = 4,n_obj=2,n_constr = 0,xl = np.array([1,1.4142135623730951,1.4142135623730951,1]),xu = np.array([3,3,3,3]))

    def _evaluate(self, x, out, *args, **kwargs):

        f = np.zeros(2)
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]
        x = [x]

        F = 10.0
        sigma = 10.0
        E = 2.0 * 1e5
        L = 200.0

        a = L * ((2 * x1) + np.sqrt(2.0) * x2 + np.sqrt(x3) + x4)
        b = ((F * L) / E) * ((2.0 / x1) + (2.0 * np.sqrt(2.0) / x2) - (2.0 * np.sqrt(2.0) / x3) + (2.0 / x4))

        f[0] = a
        f[1] = b
        out["F"] = f

class MyProblem2(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var = 4,n_obj=2,n_constr = 0,xl = np.array([1,1.4142135623730951,1.4142135623730951,1]),xu = np.array([3,3,3,3]))

    def _evaluate(self, x, out, *args, **kwargs):
        global nn1
        global nn2
        f = np.zeros(2)
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]
        x = [x]

        F = 10.0
        sigma = 10.0
        E = 2.0 * 1e5
        L = 200.0

        a = nn1.predict(x)
        b = nn2.predict(x)
        f[0] = a
        f[1] = b
        out["F"] = f

def combine(numberOfCycles):
    global nn1
    global nn2

    testProblem = MyProblem()
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

    a = 0

    while a <= x-1:
        for i in objectives:
            temp = i
            test1 = temp[0]
            test1 = [test1]
            test2 = temp[1]
            test2 = [test2]
            obj1[a]= test1
            obj2[a]= test2
        a = a +1

    print("objective one")
    print(obj1)

    print("objective two")
    print(obj2)

    print("variables")
    print(variables)

    global nn1
    global nn2
    global objectiveArchive
    global moreF
    global res_comb


    nn1 = mlp(solver='lbfgs', alpha=1e-5,hidden_layer_sizes = (15,), random_state = 1)
    nn1.fit(obj1, variables)
    nn2 = mlp(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
    nn2.fit(obj2, variables)
    problem = MyProblem2()
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

    loops = numberOfCycles
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
            a = a + 1


        nn1 = mlp(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
        nn1.fit(obj1, testVariable)
        nn2 = mlp(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
        nn2.fit(obj2, testVariable)

        print("objective function values at this point: ", objectives)
        print("length of objective function values during iteration: ", len(objectives))
        print("Neural Network One Score: ", nn1.score(obj1, testVariable))#gives out the neural network score
        print("Neural Network Two Score: ", nn1.score(obj2, testVariable))#gives out the neural network score

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
        f = np.zeros(2)
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]
        x = [x]

        F = 10.0
        sigma = 10.0
        E = 2.0 * 1e5
        L = 200.0

        a = L * ((2 * x1) + np.sqrt(2.0) * x2 + np.sqrt(x3) + x4)
        b = ((F * L) / E) * ((2.0 / x1) + (2.0 * np.sqrt(2.0) / x2) - (2.0 * np.sqrt(2.0) / x3) + (2.0 / x4))

        f[0] = a
        f[1] = b

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

