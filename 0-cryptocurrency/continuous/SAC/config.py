# 技术指标列表
TECHNICAL_INDICATORS_LIST = [
    "time",
    # "open",
    "high",
    "low",
    "close",
    "volume",
    "extent",
    "adx",
    "rsi",
    "macd",
    "macdsignal",
    "macdhist",
    "c_sma_4",  # 1h
    "c_sma_16",  # 4h
    "c_sma_32",  # 8h
    "c_sma_48",  # 12h
    "c_sma_96",  # 24h
    "c_sma_288",  # 72h
    "v_sma_4",  # 1h
    "v_sma_16",  # 4h
    "v_sma_32",  # 8h
    "v_sma_48",  # 12h
    "v_sma_96",  # 24h
    "v_sma_288",  # 72h
]

# 模型名称
modelName = "cryptocurrency-SAC"
# 模型保存路径
model_dir = "model"
# 日志保存路径
log_dir = "train_log/SAC"
# 输入集大小
prev_dim = (len(TECHNICAL_INDICATORS_LIST) - 1) * 4
# ####### 策略参数 #######
# 学习率(Learning Rate)
learning_rate_action = 2e-4
learning_rate_value1 = 2e-3
learning_rate_value2 = 2e-3
# 折扣因子(Discount Factor)
# discount_factor = 0.995
# 训练总次数(Total Training Epochs)
# total_training_epochs = 1000
# 每个Epoch的最小步数(Minimum Steps per Epoch)
# min_steps_per_epoch = 200
# 测试频率(Test Frequency)
# test_frequency = 10
