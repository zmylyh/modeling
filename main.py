import pandas as pd
from pasta.base.annotate import expression
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
season = ['s1', 's2']

combination = LpVariable.dict('comb', [str((b, p, s)) for b in block for p in plant for s in season], lowBound=0,
                              cat='Continuous')


def simulation(t):
    e1 = str(t[0])
    return str(tuple((e1, int(t[1]), t[2])))


def recon(s):
    dic = {'A': '平旱地', 'B': '梯田', 'C': '山坡地', 'D': '水浇地', 'E': '普通大棚 ', 'F': '智慧大棚'}
    return dic[s[0]]


# ljssb says that 以后要改return0
def query_data_1(b, p, table, column):
    if not table.query(f'作物编号=={p} & 地块类型 == "{recon(b)}"').empty:
        if table.query(f'作物编号=={p} & 地块类型 == "{recon(b)}"')[column].iloc[0] is None:
            return 0
        return int(table.query(f'作物编号=={p} & 地块类型 == "{recon(b)}"')[column].iloc[0])
    else:
        return 0


for b in block:
    for s in season:
        # limit the area
        expression_limit_area = 0
        for p in plant:
            if combination.get(str(simulation((b, p, s)))) is not None:
                expression_limit_area += (combination[simulation((b, p, s))])

        problem += expression_limit_area <= area_data[area_data['地块名称'] == b]['地块面积/亩'].values[0]

        # limit the plant 大白菜35，红萝卜37，白萝卜36,E1-E16,F1-F4，不能种大棚
        for i in range(16):
            j = i + 1
            area_name = f'E{j}'
            for m in range(35, 38):
                if combination.get(str(simulation((area_name, m, s)))) is not None:
                    print('ljssb1')
                    del combination[str(simulation((area_name, m, s)))]
        for i in range(4):
            j = i + 1
            area_name = f'F{j}'
            for m in range(35, 38):
                if combination.get(str(simulation((area_name, m, s)))) is not None:
                    print('ljssb2')
                    del combination[str(simulation((area_name, m, s)))]

        # limit the plant 食用菌38-41，只能第二季普通大鹏
        for i in range(38, 42):
            l = []
            for j in range(16):
                m = j + 1
                area_name = f'E{j}'
                l = [area_name]
            if combination.get(str(simulation((b, i, s)))) is not None:
                if b not in l or s == 's1':
                    print('ljssb3')
                    del combination[str(simulation((b, i, s)))]

        # limit the plant 水浇地第二季只能以上之一,大白菜35，红萝卜37，白萝卜36,D1-D8
        if recon(b) == '水浇地' and s == 's2':
            a = ((combination[str(simulation((b, 35, s)))] ==
                  area_data[area_data['地块名称'] == b]['地块面积/亩'].values[0])
                 and (combination[str(simulation((b, 37, s)))] == 0)
                 and (combination[str(simulation((b, 36, s)))] == 0))
            b1 = ((combination[str(simulation((b, 36, s)))] ==
                   area_data[area_data['地块名称'] == b]['地块面积/亩'].values[0])
                  and (combination[str(simulation((b, 35, s)))] == 0)
                  and (combination[str(simulation((b, 37, s)))] == 0))
            c = ((combination[str(simulation((b, 37, s)))] ==
                  area_data[area_data['地块名称'] == b]['地块面积/亩'].values[0])
                 and (combination[str(simulation((b, 35, s)))] == 0)
                 and (combination[str(simulation((b, 36, s)))] == 0))
            problem += a or b1 or c

        # limit the block 普通大棚第2季只能种食用菌38-41， E1-E16
        l = []
        for i in range(16):
            j = i + 1
            area_name = f'E{j}'
            l = [area_name]
        for i in plant:
            if combination.get(str(simulation((b, i, s)))) is not None:
                if b in l and i not in range(38, 42) and s == 's2':
                    print('ljssb4')
                    del combination[str(simulation((b, i, s)))]

        # limit the block 平旱地A1-A6、梯田B1-B14和山坡地C1-C6每年适宜单季种植粮食类作物（水稻除外）
        if recon(b) == '平旱地' or recon(b) == '梯田' or recon(b) == '山坡地':
            if s == 's2':
                for p in plant:
                    if combination.get(str(simulation((b, p, s)))) is not None:
                        print('ljssb5')
                        del combination[str(simulation((b, p, s)))]
            for p in plant:
                if p not in range(1, 16) and s == 's1':
                    if combination.get(str(simulation((b, p, s)))) is not None:
                        print('ljssb6')
                        del combination[str(simulation((b, p, s)))]

        # limit the block 水浇地D1-D8只能单季种植水稻，或两季蔬菜
        #if recon(b) == '水浇地' and s == 's2':

# = sum 植物price*面积*亩产量 - 植物cost*面积
# the best condition 1_1

expression = 0
for b in block:
    for p in plant:
        for s in season:
            if combination.get(str(simulation((b, p, s)))) is not None:
                expression += lpSum(query_data_1(b, p, production_data, '亩产量/斤')
                                    * combination[simulation((b, p, s))]
                                    * query_data_1(b, p, price_data, 'max')
                                    - query_data_1(b, p, cost_data, '种植成本/(元/亩)')
                                    * combination[simulation((b, p, s))])
problem += expression

problem.solve()

print("Status: ", LpStatus[problem.status])
print("Max z = ", value(problem.objective))
for v in problem.variables():
    if v.varValue != 0:
        print(f'{v.name} = {v.varValue}')
# print(f'{v.name} = {v.varValue}')
# print(production_data)
# print(block)
# print(plant)

# for b in block:
#     for p in plant:
#         if query_data_1(b, p, production_data, '亩产量/斤') is None :
#             print(b, p)
