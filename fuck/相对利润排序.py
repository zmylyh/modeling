import pandas as pd

data = pd.read_csv('csv/2_2.csv')
plant = data['作物编号']
block = data['地块类型']
cost = data['种植成本/(元/亩)']
price = data['销售单价/(元/斤)']
production = data['亩产量/斤']
min_r, max_r = [], []
for i in range(len(cost)):
    min_r.append(round(float((price[i]).split('-')[0]) * float(production[i]) - float(cost[i]), 5))
    max_r.append(round(float((price[i]).split('-')[-1]) * float(production[i]) - float(cost[i]), 5))

data['min_r'] = min_r
data['max_r'] = max_r

data.sort_values(by='min_r', ascending=False).to_csv('min_relative.csv')
data.sort_values(by='max_r', ascending=False).to_csv('max_relative.csv')
