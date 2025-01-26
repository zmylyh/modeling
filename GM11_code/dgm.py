import numpy as np

def dgm_forecast(data, forecast_steps=1, shift=1e-6):
    """
    支持数据中包含 0 的离散灰色预测模型 (DGM(1,1))
    参数:
        data (list or np.ndarray): 原始时间序列数据，长度 >= 2，允许包含 0
        forecast_steps (int): 预测未来的步数
        shift (float): 平移量，避免数据中有 0 的影响
    返回:
        forecast (list): 原始数据 + 预测值
    """
    # 数据平移，避免 0 的影响
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
    B = np.vstack([-z1, np.ones(n - 1)]).T  # 矩阵 B
    Y = np.array(data[1:])                 # 向量 Y

    # 使用最小二乘法求解 a 和 b
    params = np.linalg.pinv(B) @ Y  # 使用伪逆矩阵避免矩阵不可逆问题
    a, b = params[0], params[1]
    
    # 离散预测公式
    def predict(k):
        return (data[0] - b / a) * (1 - np.exp(a)) * np.exp(-a * (k - 1))
    
    # 生成预测值
    forecast = [data[0]]
    for k in range(2, n + forecast_steps + 1):
        forecast.append(predict(k))
    
    # 还原数据
    forecast = np.array(forecast) - shift
    return forecast


# # 测试数据
# data = [0, 10, 20, 30, 40]  # 包含 0 的数据
# forecast_steps = 5          # 预测未来 5 个点

# # 调用 DGM 函数
# result = dgm_forecast(data, forecast_steps)

# # 输出结果
# print("原始数据:", data)
# print("预测结果:", result)
