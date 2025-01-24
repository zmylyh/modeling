import numpy as np
from decimal import Decimal

def gm11_forecast(X0, forecast_steps=1):
    """
    X0: 原始数据序列（list 或 numpy array）
    forecast_steps: 预测步数
    """
    X0 = np.array(X0, dtype=float)
    # 1. 一次累加生成 X1
    X1 = np.cumsum(X0)
    # 2. 构造数据矩阵 B 和常数项 Y
    Z = 0.5 * (X1[:-1] + X1[1:])
    B = np.column_stack((-Z, np.ones(len(Z))))
    Y = X0[1:]
    # 3. 求解灰参数 a, u
    [[a], [u]] = np.linalg.lstsq(B, Y, rcond=None)[0].reshape(2,1)
    a, u = Decimal(a), Decimal(u)
    # 4. 预测函数
    def x1_hat(k):
        return (Decimal(X0[0]) - u/a) * np.exp(-a*k) + u/a
    # 5. 还原为 X0 预测值
    result = []
    for k in range(len(X0) + forecast_steps):
        if k == 0:
            result.append(X0[0])
        else:
            result.append(float(x1_hat(k) - x1_hat(k-1)))
    return result[-forecast_steps:]    
import numpy as np

data = [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
predicted = gm11_forecast(data, forecast_steps=3)
print(predicted)
import matplotlib.pyplot as plt

def plot_prediction(X0, forecast_steps=3):
    """
    利用 gm11_forecast 得到预测结果并绘图
    """
    forecast_values = gm11_forecast(X0, forecast_steps)

    # 原始序列
    plt.plot(range(len(X0)), X0, 'bo-', label='Original')

    # 预测序列
    plt.plot(range(len(X0), len(X0) + forecast_steps),
             forecast_values, 'ro--', label='Forecast')
    plt.xlabel('时间')
    plt.ylabel('数据')
    plt.title('GM(1,1) 灰色预测')
    plt.legend()
    plt.show()
plot_prediction(data)