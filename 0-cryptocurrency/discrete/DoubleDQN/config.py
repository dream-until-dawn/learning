# 技术指标列表
TECHNICAL_INDICATORS_LIST = [
    "ts",
    "close",
    "dif",
    "dea",
    "macd",
]

# 模型名称
modelName = "cryptocurrency-DoubleDQN-1D"
# 模型保存路径
model_dir = "model"
# 日志保存路径
log_dir = "train_log/DoubleDQN-1D"
# 输入集大小
prev_dim = (len(TECHNICAL_INDICATORS_LIST) - 2) * 3
# ####### 策略参数 #######
# 学习率(Learning Rate)
learning_rate_action = 2e-4
