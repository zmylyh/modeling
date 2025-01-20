import pandas as pd

# read data from csv file
data1_1 = pd.read_csv('csv/1_1.csv')
data1_2 = pd.read_csv('csv/1_2.csv')
data2_1 = pd.read_csv('csv/2_1.csv')
data2_2 = pd.read_csv('csv/2_2.csv')

def recon(s):
    dic = {'A': '平旱地', 'B': '梯田', 'C': '山坡地', 'D': '水浇地', 'E': '普通大棚 ', 'F': '智慧大棚'}
    return dic[s[0]]

def query_data_1(b, p, table, column):
    if not table.query(f'作物编号=={p} & 地块类型 == "{recon(b)}"').empty:
        if table.query(f'作物编号=={p} & 地块类型 == "{recon(b)}"')[column].iloc[0] is None:
            return 0
        return table.query(f'作物编号=={p} & 地块类型 == "{recon(b)}"')[column]
    else:
        return 0

d22 = data2_2[['作物编号','地块类型', '亩产量/斤']]
d21 = data2_1[['种植地块', '作物编号', '种植面积/亩']]

production = []
for i in range(len(d21[['种植地块', '作物编号']])):
    b, p = d21['种植地块'][i], d21['作物编号'][i]
    result = d21['种植面积/亩'][i]
    result_1 = query_data_1(b, p, d22, '亩产量/斤')
    if result_1.iloc[0] != 0:
        production.append(int(result_1.iloc[0]) * float(result))
print(production)
d21['production'] = production



data = d21[['作物编号', 'production']]
plant = set(data['作物编号'])
result_1 = []
result_2 = []
for p in plant:
    q = data.query(f'作物编号 == {p}')['production'].values
    q = q.tolist()
    prod = sum(q)
    result_1.append(p)
    result_2.append(prod)
n = pd.DataFrame({'作物编号': result_1, '产量': result_2})
n.to_csv('csv/production.csv')
