# Q(state,action) = state 下最优 action 分数 + 误差

# 为了限制误差的范围,可以对它去均值,相当于去基线

# Q(state,action) = state 下最优 action 分数 + 误差 - mean(误差)

# 这么理解起来有点蛋疼,所以我这么理解

# Q(state,action) = state 分数 + action 分数 - mean(action 分数)
