# 这种只考虑最大化 reward 而不考虑 state 的强化学习方法被称为“行为主义强化学习”或“行为主义方法”（Behaviorist Reinforcement Learning）。在这种方法中，策略（policy）只基于采取哪些动作（actions）能够获得最大化的 reward，而不考虑环境状态（state）的变化。具体来说，这种方法注重的是动作和奖励之间的直接关系，而忽略了状态的中介作用。

# 一个经典的例子是无模型强化学习（model-free reinforcement learning），其中策略学习完全依赖于直接的奖励反馈，而不是通过构建环境的模型。具体的算法有 Q-learning 和 Sarsa 等，它们都专注于通过学习状态-动作对的价值（Q 值）来最大化累积奖励。
