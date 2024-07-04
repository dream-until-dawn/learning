import torch
from torch.distributions import Normal
import pandas as pd
import numpy as np

from model import ActionModel, SACModel


class readDS:
    def __init__(self, open_file_name):
        self.open_file_name = open_file_name

    def pull_data(self, target="test") -> pd.DataFrame:
        return pd.read_csv(self.open_file_name + f"_{target}.csv")


class Player:
    def __init__(self, model_action: ActionModel):
        self.model_action = model_action
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.capital = 1e3
        self.index = 0
        self.position = 0
        self.data = readDS("DS/BTC-USDT-SWAP-15m").pull_data()
        print(self.data.shape)
        self.win_count = 0
        self.total_count = len(self.data) - 10  # 总次数

    def play(self):
        while self.index < self.total_count:
            # while self.index < 200:
            s = self.data.iloc[self.index, 3:].values.astype(np.float32)
            s_tensor = torch.FloatTensor(s).reshape(1, 24).to(self.device)
            mu, sigma = self.model_action(s_tensor)
            mu_value = mu.item()
            sigma_value = sigma.item()
            # 创建正态分布
            normal_dist = Normal(mu_value, sigma_value)
            # 生成服从正态分布的随机数，并截断在0到1之间
            a = normal_dist.sample().clamp(0, 1).item()
            ex = self.data.iloc[self.index + 1, 3:].values.astype(np.float32)[-3]
            # print(f"action: {a:.4f}\tex: {ex:.4f}")
            if a > 0.5 and ex > 0.0005:
                self.win_count += 1
            elif a < 0.5 and ex < -0.0005:
                self.win_count += 1
            if a > 0.5:
                self.capital *= 1 - 0.0005 + ex
            self.index += 1
        print(
            f"final capital: {self.capital}\tfinal win rate: {self.win_count / self.total_count:.4f}"
        )


if __name__ == "__main__":
    myModel = SACModel(mode="eval")
    myModel.loadModel()
    player = Player(myModel.model_action)

    for _ in range(1):
        player.play()
