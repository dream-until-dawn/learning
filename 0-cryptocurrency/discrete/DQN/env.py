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
        self.colse = self.df["close"]
        del self.df["close"]
        del self.df["volume"]
        self.episode = 0
        self.data_index = 0
        self.total_steps = len(self.df)
        self.starting_point = 4
        self.ending_point = self.total_steps - 4
        self.play_steps = 8640  # 3months
        print(f"df shape: {self.df.shape},colos shape: {self.colse.shape}")
        self.PS = 0
        self.UG = 0
        self.total_revenue = 1

    @property  # 当前状态
    def current_state(self) -> pd.DataFrame:
        return self.df.iloc[self.data_index]

    @property  # 下一状态
    def next_state(self) -> pd.DataFrame:
        return self.df.iloc[self.data_index + 1]

    @property  # 当前价格
    def current_price(self) -> pd.DataFrame:
        return self.colse[self.data_index]

    @property  # 下一价格
    def next_price(self) -> pd.DataFrame:
        return self.colse[self.data_index + 1]

    def get_next_state(self) -> np.ndarray:
        a1 = [self.PS] * 12
        a2 = [self.UG] * 12
        a3 = self.df.iloc[self.data_index + 1].to_numpy()
        return np.concatenate((a1, a2, a3))

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
        self.PS = 0
        self.UG = 0
        self.total_revenue = 0
        return self.get_next_state()

    def step(self, action) -> tuple[np.ndarray, float, bool, dict]:
        over = False
        # 是否已结束
        if self.data_index > self.ending_point:
            over = True
            return self.get_next_state(), 0, over, {}
        if self.total_revenue < 0.1:
            over = True
            return self.get_next_state(), -99, over, {}

        reward = 0
        if self.PS == 0:
            reward = self.noPosition(action)
        elif self.PS == 1:
            reward = self.longPosition(action)
        else:
            reward = self.shortPosition(action)

        return self.get_next_state(), reward, over, {}

    def noPosition(self, action) -> float:
        next_increase = self.next_state[-3] / 100
        if action == 1:
            if abs(next_increase) > 0.005:
                return -0.1
            else:
                return 0.1
        elif action == 2:
            self.PS = 1
            self.UG = next_increase - 0.0005
            return self.UG
        elif action == 3:
            self.PS = -1
            self.UG = -next_increase - 0.0005
            return self.UG
        else:
            return -99

    def longPosition(self, action) -> float:
        next_increase = self.next_state[-3] / 100
        if action == 1:
            self.UG = (1 + self.UG) * (1 + next_increase) - 1
            return next_increase
        elif action == 4:
            reward = self.UG - next_increase
            self.PS = 0
            self.UG = 0
            return reward
        else:
            return -99

    def shortPosition(self, action) -> float:
        next_increase = -self.next_state[-3] / 100
        if action == 1:
            self.UG = (1 + self.UG) * (1 + next_increase) - 1
            return next_increase
        elif action == 5:
            reward = self.UG - next_increase
            self.PS = 0
            self.UG = 0
            return reward
        else:
            return -99

    def random_action(self) -> int:
        if self.PS == 0:
            return np.random.choice([1, 2, 3], 1)[0]
        elif self.PS == 1:
            return np.random.choice([1, 1, 4], 1)[0]
        else:
            return np.random.choice([1, 1, 5], 1)[0]


if __name__ == "__main__":
    env = MyWrapper()
    state = env.reset()

    for i in range(10):
        action = env.random_action()
        state, reward, over, _ = env.step(action)
        print(f"当前动作: {action}\t状态: {state[-3]:.2f}\t奖励: {reward:.2f}")
        print(env.get_next_state())
