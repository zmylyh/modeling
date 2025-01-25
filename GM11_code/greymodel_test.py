import data2
import numpy as np
from greymodel import *

medal_data = data2.getMedal()
medal_data['gs'] = medal_data['Gold'] / medal_data['Total']
medal_data['ss'] = medal_data['Silver'] / medal_data['Total']
medal_data['bs'] = medal_data['Bronze'] / medal_data['Total']
team_sport = medal_data[['Team', 'Sport']]
# medal_data.to_csv('medal_data.csv', index=False)

type_dict = {
    'Gold': 'gs',
    'Silver': 'ss',
    'Bronze': 'bs'
}
comb = medal_data[['Team', 'Sport']].drop_duplicates()
comb.to_csv('comb.csv', index=False)

def getX0(team, sport, medal_type):
    # train_years = [1968, 1972, 1976, 1980, 1984]
    train_years = [2004, 2008, 2012, 2016, 2020]
    x0 = [0] * len(train_years)
    real_sum = 0.0
    real_count = 0
    for y in train_years:
        ith_year = (y - train_years[0]) // 4
        try:
            r = medal_data.query(f"(Team==@team) and (Year==@y) and (Sport==@sport)")[type_dict[medal_type]].tolist()[0]
            real_sum += r
            real_count += 1
        except:
            r = -1
        x0[ith_year] = r
    while real_count == 0:
        # 如果一个都没有，抛出错误，必须是indexerror
        # raise IndexError
        if train_years[-1] > 2000:
            raise IndexError
        del train_years[0]
        train_years.append(train_years[-1] + 4)
        for y in train_years:
            ith_year = (y - train_years[0]) // 4
            try:
                r = medal_data.query(f"(Team==@team) and (Year==@y) and (Sport==@sport)")[type_dict[medal_type]].tolist()[0]
                real_sum += r
                real_count += 1
            except:
                r = -1
            x0[ith_year] = r
    if real_sum == 0.0:
        return np.array([0] * len(train_years))
    real_average = real_sum / real_count
    # 否则填上平均值
    for i in range(len(x0)):
        if x0[i] == -1:
            x0[i] = real_average
    return np.array(x0)

# 预测单个序列, 返回值：参数和预测序列
def predictOnce(team, sport, medalType):
    x = getX0(team, sport, medalType)
    # print('input:', x)
    k = lambda_ks(x)
    i = 1
    try:
        while k == (-1, -1):
            # print(f'+{i}:', end=' ')
            k = lambda_ks(x + i)
            x = x + i
            i += 1
    except:
        pass
    x1 = sum_x1(x)
    z1 = aver_z1(x1)
    u = least_square_method(x, z1)
    # 预测
    x0_pre = prediction(u, x1, 2) - i + 1
    # 误差分析
    delta_k, pho_k = error(x - i + 1, x0_pre, u, k[0])
    # 打印信息
    # print("模型预测值为：")
    # print(x0_pre[len(x):])
    # print("相对误差为：")
    # print(delta_k)
    # print("级比误差为：")
    # print(pho_k)
    return delta_k[len(x):], x0_pre[len(x):]

def useParam(team, sport, medalType, param):
    x = getX0(team, sport, medalType)
    # print('input:', x)
    k = lambda_ks(x)
    i = 1
    try:
        while k == (-1, -1):
            # print(f'+{i}:', end=' ')
            k = lambda_ks(x + i)
            x = x + i
            i += 1
    except:
        pass
    x1 = sum_x1(x)
    z1 = aver_z1(x1)
    # u = least_square_method(x, z1)
    # 预测
    x0_pre = prediction(param, x1, 2) - i + 1
    # 误差分析
    delta_k, pho_k = error(x - i + 1, x0_pre, param, k[0])
    # 打印信息
    # print("模型预测值为：")
    # print(x0_pre[len(x):])
    # print("相对误差为：")
    # print(delta_k)
    # print("级比误差为：")
    # print(pho_k)
    return x0_pre[len(x):]

gold_result = []
silver_result = []
bronze_result = []
m = 'Gold'
# param, pre = predictOnce('Great Britain', 'Rowing', m)
for mt in ['Gold', 'Silver', 'Bronze']:
    for row in comb.itertuples():
        team = row.Team
        sport = row.Sport
        try:
            err, pre = predictOnce(team, sport, mt)
            if mt == 'Gold': gold_result.append(pre[0])
            if mt == 'Silver': silver_result.append(pre[0])
            if mt == 'Bronze': bronze_result.append(pre[0])
            print(team, '|' , sport, '|', mt)
        except IndexError:
            if mt == 'Gold': gold_result.append(-1)
            if mt == 'Silver': silver_result.append(-1)
            if mt == 'Bronze': bronze_result.append(-1)

comb['goldPre'] = gold_result
comb['silverPre'] = silver_result
comb['bronzePre'] = bronze_result
# comb.to_csv('2024pre.csv', index=False)
