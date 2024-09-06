import pandas as pd
from h5py.h5a import delete
from pulp import *

# read data from csv file
data1_1 = pd.read_csv('csv/1_1.csv')
data1_2 = pd.read_csv('csv/1_2.csv')
data2_1 = pd.read_csv('csv/2_1.csv')
data2_2 = pd.read_csv('csv/2_2.csv')

# condition
cost_data = data2_2[['作物编号', '地块类型', '种植季次', '种植成本/(元/亩)']]
production_data = data2_2[['作物编号', '地块类型', '种植季次', '亩产量/斤']]


def split(price):
    min, max = map(float, price.split('-'))
    return pd.Series([min, max])


data2_2[['min', 'max']] = data2_2['销售单价/(元/斤)'].apply(split)
price_data = data2_2[['作物编号', '地块类型', '种植季次', 'min', 'max']]

area_data = data1_1[['地块名称', '地块类型', '地块面积/亩']]
plant_data = data1_2[['作物编号', '作物名称', '作物类型']]

# create LP
problem = LpProblem("Problem", LpMaximize)

block = list(area_data['地块名称'])
plant = list(plant_data['作物编号'])
area = list(area_data['地块面积/亩'])

combination = LpVariable.dict('comb', [str((b, p)) for b in block for p in plant], lowBound=0, cat='Continuous')
print(combination["('A1', 1)"])
print(combination)
print(tuple("('A1', 1)"[1:-1].split(', '))[0])


def simulation(t):
    e1 = str(t[0])
    return tuple((e1, int(t[1])))


for b in block:
    # limit the area
    problem += (lpSum([combination[b, p] for p in plant])
                <= area_data[area_data['地块名称'] == b]['地块面积/亩'].values[0])

    # limit the plant 大白菜35，红萝卜37，白萝卜36,E1-E15,F1-F4
    for i in range(15):
        j = i + 1
        area_name = f'E{j}'
        for m in range(35,38):
            del combination[str(simulation((area_name, m)))]

# = sum 植物price*面积*亩产量 - 植物cost*面积
# the best condition 1_1
for b in block:
    for p in plant:
        problem += lpSum([int((production_data.query(f'作物编号=={p} & 地块类型 == {b}'))['亩产量/斤'].iloc[0])]
                         * combination[simulation((b, p))]
                         * int(price_data.query(f'作物编号=={p} & 地块类型 == {b}')['max'].iloc[0])
                         - int(cost_data.query(f'作物编号=={p} & 地块类型 == {b}')['种植成本/(元/亩)'].iloc[0])
                         * combination[simulation((b, p))])
    problem.solve()

print("Status: ", LpStatus[problem.status])
print("Max z = ", value(problem.objective))
for v in problem.variables():
    print(f'{v.name} = {v.varValue}')

phd = cost_data.query('作物编号 == 1 & 地块类型 == "平旱地"')
print(int(phd['种植成本/(元/亩)'].iloc[0]))
