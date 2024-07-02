# 其实就是去基线的 Actor_Critic 算法

# Actor_Critic 算法中使用 critic 模型估计 state 的价值,也就是估计 Q

# 这样估计出来的 Q 是没有去基线的,而要去基线也非常简单,target-value 即可

# 换个角度来想这个问题,target 是根据 next_state 估计出来的,value 是根据 state 估计出来的

# 所以两者的差值可以视为 action 好坏的衡量,这可以作为 actor 模型训练的依据
