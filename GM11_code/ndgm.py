import numpy as np

def ndgm_forecast(data, forecast_steps=1, shift=1e-6):
    """
    非均匀灰色模型 (NDGM) 的实现，支持处理包含 0 的数据
    参数:
        data (list or np.ndarray): 原始时间序列数据
        forecast_steps (int): 预测未来的步数
        shift (float): 数据平移量，避免 0 的影响
    返回:
        forecast (list): 原始数据 + 预测值
    """
    # 平移数据
    data = np.array(data) + shift
    n = len(data)
    if n < 2:
        raise ValueError("数据长度必须大于等于 2")
    
    # 累加生成序列
    x1 = np.cumsum(data)

    # 构造背景值序列 z^(1)
    z1 = np.zeros(n - 1)
    for i in range(1, n):
        z1[i - 1] = 0.5 * (x1[i] + x1[i - 1])
    
    # 构造 B 和 Y 矩阵
    B = np.vstack([-z1, np.ones(n - 1)]).T
    Y = np.array(data[1:])
    
    # 求解参数 a 和 b
    t = np.linalg.inv(B.T @ B) @ B.T @ Y
    a, b = t[0], t[1]
    
    # 建立预测公式
    def predict(k):
        return (data[0] - b / a) * (1 - np.exp(a)) * np.exp(-a * k)
    
    # 生成预测值
    forecast = [data[0]]
    for k in range(1, n + forecast_steps):
        forecast.append(predict(k))
    
    # 还原数据
    forecast = np.array(forecast) - shift
    return forecast[n:]


# # 测试数据
# data = [0, 10, 20, 30, 40]  # 包含 0 的数据
# forecast_steps = 5          # 预测未来 5 个点

# # 调用 NDGM 函数
# result = ndgm_forecast(data, forecast_steps)

# # 输出结果
# print("原始数据:", data)
# print("预测结果:", result)
