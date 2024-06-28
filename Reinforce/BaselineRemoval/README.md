# 计算 Q 时去除基线,基线使用 baseline 模型估计

# Q 是用蒙特卡洛法估计的,baseline 模型的 loss 就是它的计算结果和 Q 求 mse loss

# Q 在使用时减去 baseline 模型的计算结果,相当于去基线
