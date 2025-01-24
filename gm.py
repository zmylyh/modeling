import numpy as np
import pandas as pd
from decimal import *
import matplotlib.pyplot as plt
def Grade_ratio_test(X0):
    lambds = [X0[i - 1] / X0[i] for i in range(1, len(X0))]
    X_min = np.e ** (-2 / (len(X0) + 1))
    X_max = np.e ** (2 / (len(X0) + 1))
    for lambd in lambds:
        if lambd < X_min or lambd > X_max:
            print(f'{lambd} : 该数据未通过级比检验')
            return False
    print('该数据通过级比检验')
    return True
def model_train(X0_train):
    #AGO生成序列X1
    X1 = X0_train.cumsum()
    Z= (np.array([-0.5 * (X1[k - 1] + X1[k]) for k in range(1, len(X1))])).reshape(len(X1) - 1, 1)
    # 数据矩阵A、B
    A = (X0_train[1:]).reshape(len(Z), 1)
    B = np.hstack((Z, np.ones(len(Z)).reshape(len(Z), 1)))
    # 求灰参数
    a, u = np.linalg.inv(np.matmul(B.T, B)).dot(B.T).dot(A)
    u = Decimal(u[0])
    a = Decimal(a[0])
    print("灰参数a：", a, "，灰参数u：", u)
    return u,a
def model_predict(u,a,k,X0):
    predict_function =lambda k: (Decimal(X0[0]) - u / a) * np.exp(-a * k) + u / a 
    X1_hat = [float(predict_function(k)) for k in range(k)]
    X0_hat = np.diff(X1_hat)
    X0_hat = np.hstack((X1_hat[0], X0_hat))
    return X0_hat
'''
根据后验差比及小误差概率判断预测结果
:param X0_hat: 预测结果
:return:
'''
def result_evaluate(X0_hat,X0):
    S1 = np.std(X0, ddof=1)  # 原始数据样本标准差
    S2 = np.std(X0 - X0_hat, ddof=1)  # 残差数据样本标准差
    C = S2 / S1  # 后验差比
    Pe = np.mean(X0 - X0_hat)
    temp = np.abs((X0 - X0_hat - Pe)) < 0.6745 * S1    
    p = np.count_nonzero(temp) / len(X0)  # 计算小误差概率
    print("原数据样本标准差：", S1)
    print("残差样本标准差：", S2)
    print("后验差比：", C)
    print("小误差概率p：", p)
if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一（替换sans-serif字体）
    plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）
        # 原始数据X
    
    # data = pd.read_excel('./siwei_day_traffic.xlsx')
    # X=data[data['week_day']=='周五'].jam_num[:5].astype(float).values
    data = pd.read_csv('./test.csv')
    X = data['Mark'].astype(float).values
    print(X)
    # 训练集
    X_train = X[:int(len(X) * 0.5)]
    
    # 测试集
    X_test = X[int(len(X) * 0.5):]
 
    Grade_ratio_test(X_train)  # 判断模型可行性
    a,u=model_train(X_train)  # 训练
    Y_pred = model_predict(a,u,len(X),X)  # 预测
    Y_train_pred = Y_pred[:len(X_train)]
    Y_test_pred = Y_pred[len(X_train):]
    score_test = result_evaluate(Y_test_pred, X_test)  # 评估
 
    # 可视化
    plt.grid()
    plt.plot(np.arange(len(X_train)), X_train, '->')
    plt.plot(np.arange(len(X_train)), Y_train_pred, '-o')
    plt.legend(['负荷实际值', '灰色预测模型预测值'])
    plt.title('训练集')
    plt.show()
 
    plt.grid()
    plt.plot(np.arange(len(X_test)), X_test, '->')
    plt.plot(np.arange(len(X_test)), Y_test_pred, '-o')
    plt.legend(['负荷实际值', '灰色预测模型预测值'])
    plt.title('测试集')
    plt.show()