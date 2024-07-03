import numpy as np


# 计算波动率
def calculate_volatility(prices, window=20):
    # 计算收益率（只保留最后 window + 1 个价格数据）
    recent_prices = prices[-(window + 1) :]
    returns = np.diff(recent_prices) / recent_prices[:-1]
    # 计算窗口期内的波动率
    volatility = np.std(returns)
    return volatility


# 计算最大回撤
def calculate_max_drawdown(prices):
    peak = np.maximum.accumulate(prices)
    drawdown = (peak - prices) / peak
    max_drawdown = np.max(drawdown)
    return max_drawdown


# 计算贝塔值
def calculate_beta(asset_returns, market_returns):
    covariance_matrix = np.cov(asset_returns, market_returns)
    beta = covariance_matrix[0, 1] / covariance_matrix[1, 1]
    return beta


# 计算风险值
def calculate_risk(prices, window=20):
    volatility = calculate_volatility(prices, window)
    max_drawdown = calculate_max_drawdown(prices)

    # 结合多个风险指标
    total_risk = (volatility + max_drawdown) / 2
    return total_risk


if __name__ == "__main__":
    A_prices = np.array([100, 110, 121, 123, 114, 109, 106, 110, 121, 109])
    B_prices = np.array([0.1, 0.07, 0.08, -0.1, -0.1, 0.05, -0.06, -0.02, -0.01, 0.1])
    C_prices = np.array([0.1, 0.07, 0.08, -0.0, -0.1, 0.05, -0.06, -0.02, -0.01, 0.1])
    res = calculate_volatility(A_prices)
    print(res)
    res = calculate_max_drawdown(A_prices)
    print(res)
    res = calculate_beta(B_prices, C_prices)
    print(res)
