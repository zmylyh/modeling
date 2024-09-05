import pandas as pd
from pulp import *

# read data from csv file
data1_1 = pd.read_csv('1_1.csv')
data1_2 = pd.read_csv('1_2.csv')
data2_1 = pd.read_csv('2_1.csv')
data2_2 = pd.read_csv('2_2.csv')

# condition






problem = LpProblem("Problem", LpMaximize)

x1 = LpVariable("x1", 0)
x2 = LpVariable("x2", 0)

x5 = LpVariable("x5", 0)
x6 = LpVariable("x6", 0)
x3 = 0.8 * x5
x4 = 0.75 * x6

problem += 24 * x1 + 16 * x2 + 44 * x3 + 32 * x4 - 3 * x5 - 3 * x6

problem += 4 * x1 + 3 * x2 + 4 * x5 + 3 * x6 <= 600
problem += 4 * x1 + 2 * x2 + 6 * x5 + 4 * x6 <= 480
problem += x1 + x5 <= 100

problem.solve()

print("Status: ", LpStatus[problem.status])
print("Max z = ", value(problem.objective))
for v in problem.variables():
    print(f'{v.name} = {v.varValue}')