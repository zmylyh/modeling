import data
import numpy as np
from greymodel import *

medal_data = data.getMedal()
medal_data['gs'] = medal_data['Gold'] / medal_data['Total']
medal_data['ss'] = medal_data['Silver'] / medal_data['Total']
medal_data['bs'] = medal_data['Bronze'] / medal_data['Total']
team_sport = medal_data[['Team', 'Sport']]
# medal_data.to_csv('medal_data.csv', index=False)
train_years = [1968, 1972, 1976, 1980, 1984]
type_dict = {
    'Gold': 'gs',
    'Silver': 'ss',
    'Bronze': 'bs'
}
comb = medal_data[['Team', 'Sport']].drop_duplicates(keep=False)

def getX0(team, sport, medal_type):
    x0 = []
    for y in train_years:
        r = medal_data.query(f"(Year==@y) and (Team==@team) and (Sport==@sport)")[type_dict[medal_type]].tolist()[0]
        x0.append(r)
    return np.array(x0)

result_G = []
result_S = []
result_B = []

# for c in comb:
#     team = c.Team
#     sport = c.Sport
#     isSuccess = True
#     for m in ['Gold', 'Silver', 'Bronze']:
#         try:
#             x = getX0(team, sport, m)
#             k = lambda_ks(x)
#             x1 = sum_x1(x)
#             z1 = aver_z1(x1)
#             u = least_square_method(x, z1)
#             # 预测
#             x0_pre = prediction(u, x1, 8)
#             # 误差分析
#             delta_k, pho_k = error(x, x0_pre, u, k)
#             # 打印信息
#             print("模型预测值为：")
#             print(x0_pre[:len(x)])
#             print("相对误差为：")
#             print(delta_k)
#             print("级比误差为：")
#             print(pho_k)
#             print("1993年预测值为：")
#             print(x0_pre[7])
#         except:
#             isSuccess = False
#             break

team = 'Great Britain'
sport = 'Diving'
m = 'Gold'
x = getX0(team, sport, m)
k = lambda_ks(x)
x1 = sum_x1(x)
z1 = aver_z1(x1)
u = least_square_method(x, z1)
# 预测
x0_pre = prediction(u, x1, 8)
# 误差分析
delta_k, pho_k = error(x, x0_pre, u, k)
# 打印信息
print("模型预测值为：")
print(x0_pre[:len(x)])
print("相对误差为：")
print(delta_k)
print("级比误差为：")
print(pho_k)
