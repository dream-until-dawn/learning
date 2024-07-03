import random

import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np

from env import MyWrapper
from model import SACModel

import config


class Player:
    def __init__(self, model_action: nn.Sequential):
        self.env = MyWrapper()
        self.model_action = model_action
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 玩一局游戏并记录数据
    def play(self):
        data = []
        reward_sum = 0

        state = self.env.reset()
        over = False
        while not over:
            # 根据概率采样
            s_tensor = (
                torch.FloatTensor(state).reshape(1, config.prev_dim).to(self.device)
            )
            mu, sigma = self.model_action(s_tensor)
            mu_value = mu.item()
            sigma_value = sigma.item()
            # 创建正态分布
            normal_dist = Normal(mu_value, sigma_value)
            # 生成服从正态分布的随机数，并截断在0到1之间
            action = normal_dist.sample().clamp(0, 1).item()
            # print(f"action\t{action}")

            next_state, reward, over, _ = self.env.step(action)

            data.append((state, action, reward, next_state, over))
            reward_sum += reward

            state = next_state

        return data, reward_sum


if __name__ == "__main__":
    sacModel = SACModel(mode="eval")
    sacModel.loadModel()

    player = Player(sacModel.model_action)

    for _ in range(50):
        data, reward_sum = player.play()
        reward = np.array([d[2] for d in data])
        win_reward = np.array(reward > 0)
        count = np.sum(win_reward)
        print(
            f"count: {count}\twin rate: {count / len(reward)*100:.2f}%\tsum reward: {np.sum(reward)}"
        )
