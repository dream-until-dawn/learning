# 技术指标列表
TECHNICAL_INDICATORS_LIST = [
    "time",
    "close",
    "volume",
    "MACD",
    "MACD信号线",
    "RSI",
    "band-Upper",
    "band-Lower",
    "VWAP",
    "ATR",
    "%K",
    "%D",
    "volRatio_5",
    "VPR",
    "SMA_c4",
    "SMA_c16",
    "SMA_c48",
    "SMA_c96",
    "SMA_c288",
    "SMA_v4",
    "SMA_v16",
    "SMA_v48",
    "SMA_v96",
    "SMA_v288",
    "涨幅",
    "振幅",
    "量涨幅",
]

# 模型名称
modelName = "cryptocurrency-PPO"
# 模型保存路径
model_dir = "model"
# 日志保存路径
log_dir = "train_log/PPO"
# 输入集大小
prev_dim = (len(TECHNICAL_INDICATORS_LIST) - 3) * 4
# ####### 策略参数 #######
# 学习率(Learning Rate)
learning_rate_action = 5e-4
learning_rate_value = 5e-3
