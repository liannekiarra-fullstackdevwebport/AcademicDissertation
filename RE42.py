#this code combines the optimisation algorithm and the nerual networks and gives the output.

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


class RE42(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var=6, n_obj=4, n_constr=0, xl=np.array([150,20,13,10,14,0.63]), xu=np.array([274.32,32.31,25, 11.71, 18,0.75]),
                             )

    def _evaluate(self, x, out, *args, **kwargs):
        f = np.zeros(4)
        # NOT g
        constraintFuncs = np.zeros(9)

        x_L = x[0]
        x_B = x[1]
        x_D = x[2]
        x_T = x[3]
        x_Vk = x[4]
        x_CB = x[5]

        displacement = 1.025 * x_L * x_B * x_T * x_CB
        V = 0.5144 * x_Vk
        g = 9.8065
        Fn = V / np.power(g * x_L, 0.5)
        a = (4977.06 * x_CB * x_CB) - (8105.61 * x_CB) + 4456.51
        b = (-10847.2 * x_CB * x_CB) + (12817.0 * x_CB) - 6960.32

        power = (np.power(displacement, 2.0 / 3.0) * np.power(x_Vk, 3.0)) / (a + (b * Fn))
        outfit_weight = 1.0 * np.power(x_L, 0.8) * np.power(x_B, 0.6) * np.power(x_D, 0.3) * np.power(x_CB, 0.1)
        steel_weight = 0.034 * np.power(x_L, 1.7) * np.power(x_B, 0.7) * np.power(x_D, 0.4) * np.power(x_CB, 0.5)
        machinery_weight = 0.17 * np.power(power, 0.9)
        light_ship_weight = steel_weight + outfit_weight + machinery_weight

        ship_cost = 1.3 * ((2000.0 * np.power(steel_weight, 0.85)) + (3500.0 * outfit_weight) + (
                2400.0 * np.power(power, 0.8)))
        capital_costs = 0.2 * ship_cost

        DWT = displacement - light_ship_weight

        running_costs = 40000.0 * np.power(DWT, 0.3)

        round_trip_miles = 5000.0
        sea_days = (round_trip_miles / 24.0) * x_Vk
        handling_rate = 8000.0

        daily_consumption = ((0.19 * power * 24.0) / 1000.0) + 0.2
        fuel_price = 100.0
        fuel_cost = 1.05 * daily_consumption * sea_days * fuel_price
        port_cost = 6.3 * np.power(DWT, 0.8)

        fuel_carried = daily_consumption * (sea_days + 5.0)
        miscellaneous_DWT = 2.0 * np.power(DWT, 0.5)

        cargo_DWT = DWT - fuel_carried - miscellaneous_DWT
        port_days = 2.0 * ((cargo_DWT / handling_rate) + 0.5)
        RTPA = 350.0 / (sea_days + port_days)

        voyage_costs = (fuel_cost + port_cost) * RTPA
        annual_costs = capital_costs + running_costs + voyage_costs
        annual_cargo = cargo_DWT * RTPA

        f[0] = annual_costs / annual_cargo
        f[1] = light_ship_weight
        # f_2 is dealt as a minimization problem
        f[2] = -annual_cargo

        # Reformulated objective functions
        constraintFuncs[0] = (x_L / x_B) - 6.0
        constraintFuncs[1] = -(x_L / x_D) + 15.0
        constraintFuncs[2] = -(x_L / x_T) + 19.0
        constraintFuncs[3] = 0.45 * np.power(DWT, 0.31) - x_T
        constraintFuncs[4] = 0.7 * x_D + 0.7 - x_T
        constraintFuncs[5] = 500000.0 - DWT
        constraintFuncs[6] = DWT - 3000.0
        constraintFuncs[7] = 0.32 - Fn

        KB = 0.53 * x_T
        BMT = ((0.085 * x_CB - 0.002) * x_B * x_B) / (x_T * x_CB)
        KG = 1.0 + 0.52 * x_D
        constraintFuncs[8] = (KB + BMT - KG) - (0.07 * x_B)

        constraintFuncs = np.where(constraintFuncs < 0, -constraintFuncs, 0)
        f[3] = constraintFuncs[0] + constraintFuncs[1] + constraintFuncs[2] + constraintFuncs[3] + constraintFuncs[4] + \
               constraintFuncs[5] + constraintFuncs[6] + constraintFuncs[7] + constraintFuncs[8]

        out["F"] = f
        out["G"] = g

class RE42RandomiseData(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var=6, n_obj=4, n_constr=0, xl=np.array([150,20,13,10,14,0.63]), xu=np.array([274.32,32.31,25, 11.71, 18,0.75]),
                             )

    def _evaluate(self, x, out, *args, **kwargs):
        f = np.zeros(4)
        # NOT g
        constraintFuncs = np.zeros(9)

        x_L = x[0]
        x_B = x[1]
        x_D = x[2]
        x_T = x[3]
        x_Vk = x[4]
        x_CB = x[5]

        displacement = 1.025 * x_L * x_B * x_T * x_CB
        V = 0.5144 * x_Vk
        g = 9.8065
        Fn = V / np.power(g * x_L, 0.5)
        a = (4977.06 * x_CB * x_CB) - (8105.61 * x_CB) + 4456.51
        b = (-10847.2 * x_CB * x_CB) + (12817.0 * x_CB) - 6960.32

        power = (np.power(displacement, 2.0 / 3.0) * np.power(x_Vk, 3.0)) / (a + (b * Fn))
        outfit_weight = 1.0 * np.power(x_L, 0.8) * np.power(x_B, 0.6) * np.power(x_D, 0.3) * np.power(x_CB, 0.1)
        steel_weight = 0.034 * np.power(x_L, 1.7) * np.power(x_B, 0.7) * np.power(x_D, 0.4) * np.power(x_CB, 0.5)
        machinery_weight = 0.17 * np.power(power, 0.9)
        light_ship_weight = steel_weight + outfit_weight + machinery_weight

        ship_cost = 1.3 * ((2000.0 * np.power(steel_weight, 0.85)) + (3500.0 * outfit_weight) + (
                2400.0 * np.power(power, 0.8)))
        capital_costs = 0.2 * ship_cost

        DWT = displacement - light_ship_weight

        running_costs = 40000.0 * np.power(DWT, 0.3)

        round_trip_miles = 5000.0
        sea_days = (round_trip_miles / 24.0) * x_Vk
        handling_rate = 8000.0

        daily_consumption = ((0.19 * power * 24.0) / 1000.0) + 0.2
        fuel_price = 100.0
        fuel_cost = 1.05 * daily_consumption * sea_days * fuel_price
        port_cost = 6.3 * np.power(DWT, 0.8)

        fuel_carried = daily_consumption * (sea_days + 5.0)
        miscellaneous_DWT = 2.0 * np.power(DWT, 0.5)

        cargo_DWT = DWT - fuel_carried - miscellaneous_DWT
        port_days = 2.0 * ((cargo_DWT / handling_rate) + 0.5)
        RTPA = 350.0 / (sea_days + port_days)

        voyage_costs = (fuel_cost + port_cost) * RTPA
        annual_costs = capital_costs + running_costs + voyage_costs
        annual_cargo = cargo_DWT * RTPA

        f[0] = annual_costs / annual_cargo
        f[1] = light_ship_weight
        # f_2 is dealt as a minimization problem
        f[2] = -annual_cargo

        # Reformulated objective functions
        constraintFuncs[0] = (x_L / x_B) - 6.0
        constraintFuncs[1] = -(x_L / x_D) + 15.0
        constraintFuncs[2] = -(x_L / x_T) + 19.0
        constraintFuncs[3] = 0.45 * np.power(DWT, 0.31) - x_T
        constraintFuncs[4] = 0.7 * x_D + 0.7 - x_T
        constraintFuncs[5] = 500000.0 - DWT
        constraintFuncs[6] = DWT - 3000.0
        constraintFuncs[7] = 0.32 - Fn

        KB = 0.53 * x_T
        BMT = ((0.085 * x_CB - 0.002) * x_B * x_B) / (x_T * x_CB)
        KG = 1.0 + 0.52 * x_D
        constraintFuncs[8] = (KB + BMT - KG) - (0.07 * x_B)

        constraintFuncs = np.where(constraintFuncs < 0, -constraintFuncs, 0)
        f[3] = constraintFuncs[0] + constraintFuncs[1] + constraintFuncs[2] + constraintFuncs[3] + constraintFuncs[4] + \
               constraintFuncs[5] + constraintFuncs[6] + constraintFuncs[7] + constraintFuncs[8]

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

    testProblem = RE42RandomiseData()
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



    a = 0

    while a <= x-1:
        for i in objectives:
            temp = i
            test1 = temp[0]
            test1 = [test1]
            test2 = temp[1]
            test2 = [test2]
            test3 = temp[2]
            test3 = [test3]
            test4 = temp[3]
            test4 = [test4]
            obj1[a]= test1
            obj2[a]= test2
            obj3[a] = test3
            obj4[a] = test4
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


    problem = RE42()
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

                obj3[a] = test3
                obj4[a] = test4

        a = a + 1


        nn1 = mlp(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
        nn1.fit(obj1, testVariable)
        nn2 = mlp(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
        nn2.fit(obj2, testVariable)
        nn3 = mlp(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
        nn3.fit(obj3, testVariable)
        nn4 = mlp(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
        nn4.fit(obj4, testVariable)



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
    def evaluate(self, x):
        f = np.zeros(self.n_objectives)
        # NOT g
        constraintFuncs = np.zeros(self.n_original_constraints)

        x_L = x[0]
        x_B = x[1]
        x_D = x[2]
        x_T = x[3]
        x_Vk = x[4]
        x_CB = x[5]

        displacement = 1.025 * x_L * x_B * x_T * x_CB
        V = 0.5144 * x_Vk
        g = 9.8065
        Fn = V / np.power(g * x_L, 0.5)
        a = (4977.06 * x_CB * x_CB) - (8105.61 * x_CB) + 4456.51
        b = (-10847.2 * x_CB * x_CB) + (12817.0 * x_CB) - 6960.32

        power = (np.power(displacement, 2.0 / 3.0) * np.power(x_Vk, 3.0)) / (a + (b * Fn))
        outfit_weight = 1.0 * np.power(x_L, 0.8) * np.power(x_B, 0.6) * np.power(x_D, 0.3) * np.power(x_CB, 0.1)
        steel_weight = 0.034 * np.power(x_L, 1.7) * np.power(x_B, 0.7) * np.power(x_D, 0.4) * np.power(x_CB, 0.5)
        machinery_weight = 0.17 * np.power(power, 0.9)
        light_ship_weight = steel_weight + outfit_weight + machinery_weight

        ship_cost = 1.3 * ((2000.0 * np.power(steel_weight, 0.85)) + (3500.0 * outfit_weight) + (
                2400.0 * np.power(power, 0.8)))
        capital_costs = 0.2 * ship_cost

        DWT = displacement - light_ship_weight

        running_costs = 40000.0 * np.power(DWT, 0.3)

        round_trip_miles = 5000.0
        sea_days = (round_trip_miles / 24.0) * x_Vk
        handling_rate = 8000.0

        daily_consumption = ((0.19 * power * 24.0) / 1000.0) + 0.2
        fuel_price = 100.0
        fuel_cost = 1.05 * daily_consumption * sea_days * fuel_price
        port_cost = 6.3 * np.power(DWT, 0.8)

        fuel_carried = daily_consumption * (sea_days + 5.0)
        miscellaneous_DWT = 2.0 * np.power(DWT, 0.5)

        cargo_DWT = DWT - fuel_carried - miscellaneous_DWT
        port_days = 2.0 * ((cargo_DWT / handling_rate) + 0.5)
        RTPA = 350.0 / (sea_days + port_days)

        voyage_costs = (fuel_cost + port_cost) * RTPA
        annual_costs = capital_costs + running_costs + voyage_costs
        annual_cargo = cargo_DWT * RTPA

        f[0] = annual_costs / annual_cargo
        f[1] = light_ship_weight
        # f_2 is dealt as a minimization problem
        f[2] = -annual_cargo

        # Reformulated objective functions
        constraintFuncs[0] = (x_L / x_B) - 6.0
        constraintFuncs[1] = -(x_L / x_D) + 15.0
        constraintFuncs[2] = -(x_L / x_T) + 19.0
        constraintFuncs[3] = 0.45 * np.power(DWT, 0.31) - x_T
        constraintFuncs[4] = 0.7 * x_D + 0.7 - x_T
        constraintFuncs[5] = 500000.0 - DWT
        constraintFuncs[6] = DWT - 3000.0
        constraintFuncs[7] = 0.32 - Fn

        KB = 0.53 * x_T
        BMT = ((0.085 * x_CB - 0.002) * x_B * x_B) / (x_T * x_CB)
        KG = 1.0 + 0.52 * x_D
        constraintFuncs[8] = (KB + BMT - KG) - (0.07 * x_B)

        constraintFuncs = np.where(constraintFuncs < 0, -constraintFuncs, 0)
        f[3] = constraintFuncs[0] + constraintFuncs[1] + constraintFuncs[2] + constraintFuncs[3] + constraintFuncs[4] + \
               constraintFuncs[5] + constraintFuncs[6] + constraintFuncs[7] + constraintFuncs[8]

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


