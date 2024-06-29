import gym

from matplotlib import pyplot as plt


# 定义环境
class MyWrapper(gym.Wrapper):

    def __init__(self):
        env = gym.make("Pendulum-v1", render_mode="rgb_array")
        super().__init__(env)
        self.env = env
        self.step_n = 0
        plt.ion()  # 启用交互模式
        plt.figure(figsize=(3, 3))

    def reset(self):
        state, _ = self.env.reset()
        self.step_n = 0
        return state

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        over = terminated or truncated

        # 偏移reward,便于训练
        reward = (reward + 8) / 8

        # 限制最大步数
        self.step_n += 1
        if self.step_n >= 200:
            over = True

        return state, reward, over, info

    # 打印游戏图像
    def show(self):
        plt.imshow(self.env.render())
        plt.draw()  # 更新图像
        plt.pause(0.1)  # 暂停0.1秒，确保图像显示


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
        env.show()
