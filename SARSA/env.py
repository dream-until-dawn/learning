import gym


# 定义环境
# SFFF
# FHFH
# FFFH
# HFFG
# 0: left, 1: down, 2: right, 3: up
class MyWrapper(gym.Wrapper):

    def __init__(self):
        # is_slippery控制会不会滑
        env = gym.make("FrozenLake-v1", render_mode="rgb_array", is_slippery=False)

        super().__init__(env)
        self.env = env

    def reset(self):
        state, _ = self.env.reset()
        return state

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        over = terminated or truncated

        # 走一步扣一份,逼迫机器人尽快结束游戏
        if not over:
            reward = -1

        # 掉坑扣100分
        if over and reward == 0:
            reward = -100

        return state, reward, over, info

    # 打印游戏图像
    def show(self):
        from matplotlib import pyplot as plt

        plt.figure(figsize=(3, 3))
        plt.imshow(self.env.render())
        plt.show()


if __name__ == "__main__":
    env = MyWrapper()
    state = env.reset()
    over = False
    print("start")
    while not over:
        action = env.env.action_space.sample()
        next_state, reward, over, info = env.step(action)
        print(f"state: {state}, action: {action}, reward: {reward}, over: {over}")
        state = next_state
    # env.show()
