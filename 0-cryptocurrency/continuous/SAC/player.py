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


# 数据池
class Pool:

    def __init__(self, player: Player):
        self.pool = []
        self.player = player
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.pool)

    def __getitem__(self, i):
        return self.pool[i]

    # 更新动作池
    def update(self):
        self.pool = self.player.play()[0]

    # 获取一批数据样本
    def sample(self):
        # data = random.sample(self.pool, 512)
        data = self.pool

        state = (
            torch.FloatTensor(np.array([i[0] for i in data]))
            .reshape(-1, config.prev_dim)
            .to(self.device)
        )
        action = (
            torch.FloatTensor(np.array([i[1] for i in data]))
            .reshape(-1, 1)
            .to(self.device)
        )
        reward = (
            torch.FloatTensor(np.array([i[2] for i in data]))
            .reshape(-1, 1)
            .to(self.device)
        )
        next_state = (
            torch.FloatTensor(np.array([i[3] for i in data]))
            .reshape(-1, config.prev_dim)
            .to(self.device)
        )
        over = (
            torch.LongTensor(np.array([i[4] for i in data]))
            .reshape(-1, 1)
            .to(self.device)
        )

        return state, action, reward, next_state, over


if __name__ == "__main__":
    sacModel = SACModel()

    model_action = sacModel.model_action

    player = Player(model_action)
    pool = Pool(player)

    pool.update()
    state, action, reward, next_state, over = pool.sample()
    print(state.shape, action.shape, reward.shape, next_state.shape, over.shape)
