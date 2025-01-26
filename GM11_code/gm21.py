import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def gm21_predict(data, forecast_steps=10):
    # 步骤 1：原始数据序列
    x0 = np.array(data)  # 原始序列

    shift = 0
    lambdas = x0[:-1] / x0[1:]
    
    while not np.all((lambdas > 0.5) & (lambdas < 1.5)):
        shift += 1
        x0 += shift
        lambdas = x0[:-1] / x0[1:]
    
    # 步骤 2：一次累加生成（AGO）
    x1 = np.cumsum(x0)  # 累加生成的序列

    # 步骤 3：构造背景值（即序列 x1 的均值）
    z1 = (x1[:-1] + x1[1:]) / 2

    # 步骤 4：建立模型参数矩阵
    # 构造 B 矩阵
    B = np.vstack([-z1, np.ones(len(z1))]).T
    Y = x0[1:]

    # 步骤 5：用最小二乘法求解参数 a 和 b
    # (B.T * B) * theta = B.T * Y
    theta = np.linalg.inv(B.T @ B) @ B.T @ Y
    a, b = theta[0], theta[1]

    # 步骤 6：预测得到的二阶累加序列
    # 预测的累加序列（x1）
    x1_predict = np.zeros(forecast_steps)
    x1_predict[0] = x1[-1]
    
    for k in range(1, forecast_steps):
        x1_predict[k] = (x1_predict[k-1] - b/a) * np.exp(-a * (k+1)) + b/a

    # 步骤 7：还原预测值（逆累加生成 IAGO）
    x0_predict = np.diff(x1_predict)

    # 步骤 8：返回预测结果
    predicted_values = np.concatenate(([x0[0]], x0_predict))  # 合并第一项与预测数据

    if shift != 0:
        predicted_values -= shift
    # print(len(predicted_values))
    return predicted_values# , x1_predict, x0_predict

# # 示例数据：一个简单的时间序列
# data = [100, 120, 150, 180, 210, 250, 300, 360, 430]

# # 进行 GM(2,1) 预测
# predicted_values, x1_predict, x0_predict = gm21_predict(data, forecast_steps=10)

# # 打印预测结果
# print("原始数据:", data)
# print("预测结果:", predicted_values)

# # 绘制图形
# plt.plot(range(1, len(data)+1), data, 'bo-', label='原始数据')
# plt.plot(range(len(data), len(data)+10), predicted_values[len(data):], 'ro-', label='预测数据')
# plt.legend()
# plt.title('GM(2,1) 预测')
# plt.xlabel('时间')
# plt.ylabel('值')
# plt.show()
