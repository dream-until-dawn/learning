import gym
import pandas as pd
import numpy as np
import random
import time


class readDS:
    def __init__(self, open_file_name):
        self.open_file_name = open_file_name

    def pull_data(self, target="train") -> pd.DataFrame:
        return pd.read_csv(self.open_file_name + f"_{target}.csv")


# 定义环境
class MyWrapper(gym.Wrapper):

    def __init__(self, open_file_name="DS/BTC-USDT-SWAP-15m", target="train"):
        self.read_ds = readDS(open_file_name)
        self.df = self.read_ds.pull_data(target)
        self.df = self.df.set_index("time")
        self.episode = 0
        self.data_index = 0
        self.total_steps = len(self.df)
        self.starting_point = 4
        self.ending_point = self.total_steps - 4
        self.play_steps = 512 - 2

    @property  # 当前轮已执行步数
    def step_number(self) -> int:
        return self.data_index - self.starting_point

    @property  # 当前状态
    def current_state(self) -> pd.DataFrame:
        return self.df.iloc[self.data_index - 3 : self.data_index + 1]

    @property  # 下一状态
    def next_state(self) -> pd.DataFrame:
        return self.df.iloc[self.data_index - 2 : self.data_index + 2]

    # 设置随机种子
    def seed(self, seed=None) -> None:
        if seed is None:
            seed = int(round(time.time() * 1000))
        random.seed(seed)

    def reset(self) -> np.ndarray:
        self.episode += 1
        self.seed()
        self.starting_point = random.choice(
            range(4, self.total_steps - self.play_steps - 4)
        )
        self.ending_point = self.starting_point + self.play_steps
        self.data_index = self.starting_point
        # print(
        #     f"start: {self.starting_point}\tend: {self.ending_point}\ttotal: {self.total_steps}"
        # )
        return self.current_state.to_numpy()

    def step(self, action) -> tuple[np.ndarray, float, bool, dict]:
        over = False
        # 是否已结束
        if self.data_index > self.ending_point:
            over = True
        # print(f"action: {action}")
        # 反转动作
        reverse_action = 1 - action
        # 下一状态幅度
        next_extent = self.next_state.iloc[-1]["extent"] * 100
        # 计算奖励
        reward = action * next_extent
        reverse_reward = reverse_action * next_extent * -1
        # 步数+1
        self.data_index += 1

        return self.next_state.to_numpy(), reward + reverse_reward, over, {}


if __name__ == "__main__":
    env = MyWrapper()
    state = env.reset()

    for i in range(100):
        state, reward, over, _ = env.step(random.random())
