import pandas as pd
from pulp import *
import re

# read data from csv file
data1_1 = pd.read_csv('csv/1_1.csv')
data1_2 = pd.read_csv('csv/1_2.csv')
data2_1 = pd.read_csv('csv/2_1.csv')
data2_2 = pd.read_csv('csv/2_2.csv')
data_p = pd.read_csv('csv/production.csv')

# condition
cost_data = data2_2[['作物编号', '地块类型', '种植季次', '种植成本/(元/亩)']]
production_data = data2_2[['作物编号', '地块类型', '种植季次', '亩产量/斤']]
sales_data = data2_1[['种植地块', '作物编号', '种植面积/亩', '种植季次']]
production1_data = data_p[['作物编号', '产量']]


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
season = ['a0', 'a1', 'a2',
          'b0', 'b1', 'b2',
          'c0', 'c1', 'c2',
          'd0', 'd1', 'd2',
          'e0', 'e1', 'e2',
          'f0', 'f1', 'f2',
          'g0', 'g1', 'g2']

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

        # 1 limit the plant 大白菜35，红萝卜37，白萝卜36,E1-E16,F1-F4，不能种大棚
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

        # 2 limit the plant 食用菌38-41，只能第二季普通大棚
        for i in range(38, 42):
            l = []
            for j in range(16):
                m = j + 1
                area_name = f'E{j}'
                l = [area_name]
            if combination.get(str(simulation((b, i, s)))) is not None:
                if b not in l or s[-1] == '1':
                    print('ljssb3')
                    del combination[str(simulation((b, i, s)))]

        # 3 limit the block 水浇地第二季只能以上之一,大白菜35，红萝卜37，白萝卜36,D1-D8
        if recon(b) == '水浇地' and s[-1] == '2':
            for p in plant:
                if combination.get(str(simulation((b, p, s)))) is not None:
                    if p not in range(35, 38):
                        print('fix1')
                        del combination[str(simulation((b, p, s)))]
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

        # 4 limit the block 普通大棚第2季只能种食用菌38-41， E1-E16
        for i in plant:
            if combination.get(str(simulation((b, i, s)))) is not None:
                if (recon(b) == '普通大棚 ') and (i not in range(38, 42)) and (s[-1] == '2'):
                    print('ljssb4')
                    del combination[str(simulation((b, i, s)))]

        # 5 limit the block 平旱地A1-A6、梯田B1-B14和山坡地C1-C6每年适宜单季种植粮食类作物（水稻除外）
        if recon(b) == '平旱地' or recon(b) == '梯田' or recon(b) == '山坡地':
            if s[-1] == '2':
                for p in plant:
                    if combination.get(str(simulation((b, p, s)))) is not None:
                        print('ljssb5')
                        del combination[str(simulation((b, p, s)))]
            for p in plant:
                if p not in range(1, 16) and s[-1] == '0':
                    if combination.get(str(simulation((b, p, s)))) is not None:
                        print('ljssb6')
                        del combination[str(simulation((b, p, s)))]

        # 6 limit the block 水浇地D1-D8只能单季种植水稻16，或两季蔬菜17-37
        if recon(b) == '水浇地':
            for p in plant:
                if p != 16 and p not in range(17, 38):
                    if combination.get(str(simulation((b, p, s)))) is not None:
                        print('ljssb7')
                        del combination[str(simulation((b, p, s)))]
                if p == 16 and s[-1] == '2':
                    if combination.get(str(simulation((b, p, s)))) is not None:
                        print('ljssb8')
                        del combination[str(simulation((b, p, s)))]
                if p in range(17, 38):
                    if combination.get(str(simulation((b, p, s)))) is not None:
                        if s[-1] == "0":
                            a = ((combination[str(simulation((b, p, s)))] <=
                                  area_data[area_data['地块名称'] == b]['地块面积/亩'].values[0])
                                 - (combination[str(simulation((b, 16, s)))]))
                            problem += a

        # 7 limit the plant 水稻只能种在水浇地
        if p == 16 and recon(b) != '水浇地':
            for p in plant:
                if combination.get(str(simulation((b, p, s)))) is not None:
                    print('ljssb9')
                    del combination[str(simulation((b, p, s)))]

        # 8 limit the plant 35-37只能种水浇地第二季s2
        for p in range(35, 38):
            if combination.get(str(simulation((b, p, s)))) is not None:
                if recon(b) != '水浇地' or s[-1] != '2':
                    print('ljssb10')
                    del combination[str(simulation((b, p, s)))]

# 9 不能连种
for b in block:
    for p in plant:
        for s1 in season:
            for s2 in season:
                if combination.get(str(simulation((b, p, s1)))) is not None and combination.get(str(simulation((b, p, s2)))) is not None:
                    if ord(s1[0]) - ord(s2[0]) == 1:
                        problem += (combination[str(simulation((b, p, s1)))] + combination[
                            str(simulation((b, p, s2)))] <= area_data[area_data['地块名称'] == b]['地块面积/亩'].values[
                                        0])
                
# 10 豆类限制,1-5,17-19
for b in block:
    for s1 in range(6, 21):
        for number in range(3):
            total_1 = total_2 = total_3 = lpSum(0)
            for p in range(1, 6) or range(17, 20):
                if (combination.get(str(simulation((b, p, chr(ord(season[s1][0]) - 2) + str(number))))) is not None) and \
                (combination.get(str(simulation((b, p, chr(ord(season[s1][0]) - 1) + str(number))))) is not None) and \
                (combination.get(str(simulation((b, p, season[s1])))) is not None):
                    total_1 += lpSum(combination[str(simulation((b, p, season[s1])))]) 
                    total_2 += lpSum(combination[str(simulation((b, p, chr(ord(season[s1][0]) - 1) + str(number))))])
                    total_3 += lpSum(combination[str(simulation((b, p, chr(ord(season[s1][0]) - 2) + str(number))))])
            # print(type(total_1))
            area = area_data[area_data['地块名称'] == b]['地块面积/亩'].values[0]
            ex1 = (total_1 == area)
            ex2 = (total_2 == area)
            ex3 = (total_3 == area)
            
            problem += (ex1) or (ex2) or (ex3)
            # problem += ex1
            # if season[s1][1] == '0':
            #     for s in range((ord(season[s1][0])-2), (ord(season[s1][0]))):
            #         if combination.get(str(simulation((b, p, s)))):

            # = sum 植物price*面积*亩产量 - 植物cost*面积 + 植物price*(sum(亩产*今年面积)-sum(亩产*23面积))
            # the best condition 1_1

            # def query_sales_data(b, p, s, table, col):
            #     time = {"单季": "s1", "第一季": "s1", "第二季": "s2"}
            #     if not table.query(f'种植地块=={p} & 地块类型 == "{recon(b)}" & 种植季次 == {time[s]}').empty:
            #         if table.query(f'种植地块=={p} & 地块类型 == "{recon(b)}" & 种植季次 == {time[s]}')[col].iloc[0] is None:
            #             return 0
            #         return int(table.query(f'种植地块=={p} & 地块类型 == "{recon(b)}" & 种植季次 == {time[s]}')[col].iloc[0])
            #     else:
            #         return 0


def current_production(p):
    expression_1 = 0
    for b_ in block:
        for s_ in season:
            if combination.get(str(simulation((b_, p, s_)))) is not None:
                expression_1 += query_data_1(b_, p, production_data, '亩产量/斤') * combination[simulation((b_, p, s_))]
    # print(expression_1)            
    return expression_1

def year_production(p, year):
    yp = 0
    for b in block:
        current_season = [s for s in season if s[0] == chr(year + 96)]
        for s in current_season:
            if combination.get(str(simulation((b, p, s)))) is not None:
                mul = 1.1 ** year
                yp += query_data_1(b, p, production_data, '亩产量/斤') * combination[simulation((b, p, s))] * mul
    return yp

expression = 0
total_cost = 0
sale_change = 1.05
prod_change = 1.1
price_change = 1.05
cost_change = 1.05
for b in block:
    for p in plant:
        for s in season:
            if combination.get(str(simulation((b, p, s)))) is not None:
                year = (ord(s[0]) - 97) + 1 # ith year
                # 增长速率调整
                if p == 6 or p == 7:
                    sale_change = 1.1 ** (year - 1)
                else:
                    sale_change = 1.05
                prod_change = 1.1 ** (year - 1)
                cost_change = 1.05 ** (year - 1)
                if p in range(17, 38):
                    # 蔬菜价格上升5%
                    price_change = 1.05 ** (year - 1)
                elif p in range(38, 42):
                    # 食用菌价格下降5%
                    price_change = 0.95 ** (year - 1)
                total_cost += query_data_1(b, p, cost_data, '种植成本/(元/亩)') * cost_change \
                                * combination[simulation((b, p, s))]
                # expression += lpSum(query_data_1(b, p, production_data, '亩产量/斤')
                #                     * combination[simulation((b, p, s))]
                #                     * query_data_1(b, p, price_data, 'max') * price_change
                #                     - query_data_1(b, p, cost_data, '种植成本/(元/亩)') * cost_change
                #                     * combination[simulation((b, p, s))])

total_sale = 0
for s in season:
    y = (ord(s[0]) - 97) + 1
    for p in plant:
        if p == 6 or p == 7:
            sale_change = 1.1 ** (year - 1)
        else:
            sale_change = 1.05
        if p in range(17, 38):
            # 蔬菜价格上升5%
            price_change = 1.05 ** (year - 1)
        elif p in range(38, 42):
            # 食用菌价格下降5%
            price_change = 0.95 ** (year - 1)

        sale_var = LpVariable(f'sale_var_{p}', 0)
        predict_sale = int(data_p.query(f'作物编号 == {p}')['产量'].iloc[0]) * sale_change
        a = (year_production(p, y) <= predict_sale and sale_var == year_production(p, y))
        b = (year_production(p, y) >= predict_sale and sale_var == predict_sale)
        problem += a or b
        total_sale += sale_var * float(price_data.query(f'作物编号 == {p}')['max'].iloc[0]) * price_change
expression += total_sale - total_cost

problem += expression
solver = PULP_CBC_CMD(threads=32)
problem.solve(solver)

print("Status: ", LpStatus[problem.status])
print("Max z = ", value(problem.objective))


def match_pattern(s):
    m = re.match(r"comb_\('(\w+)',_(\d+),_'(\w+)'\)", s)
    return m.group(1), int(m.group(2)), m.group(3)


block = []
plant = []
for v in problem.variables():
    if v.varValue != 0:
        print(f'{v.name} = {v.varValue}')
        block.append(match_pattern(v.name)[0])
        plant.append(match_pattern(v.name)[1])
df = pd.DataFrame({'地块名称': block, '作物编号': plant})
df.to_csv('csv/result_2.csv')
