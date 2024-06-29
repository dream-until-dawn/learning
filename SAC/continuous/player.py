import time
import random

from IPython import display
import torch
import torch.nn as nn
import numpy as np

from env import MyWrapper
from model import DQNModel


class Player:
    def __init__(self, model_action: nn.Sequential):
        self.env = MyWrapper()
        self.model_action = model_action
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 玩一局游戏并记录数据
    def play(self, show=False):
        data = []
        reward_sum = 0

        state = self.env.reset()
        over = False
        while not over:
            # 根据概率采样
            s_tensor = torch.FloatTensor(state).reshape(1, 3).to(self.device)
            mu, sigma = self.model_action(s_tensor)
            action = [random.normalvariate(mu=mu.item(), sigma=sigma.item())]

            next_state, reward, over, _ = self.env.step(action)

            data.append((state, action, reward, next_state, over))
            reward_sum += reward

            state = next_state

            if show:
                display.clear_output(wait=True)
                self.env.show()

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
        # 每次更新不少于N条新数据
        old_len = len(self.pool)
        while len(self.pool) - old_len < 200:
            self.pool.extend(self.player.play()[0])

        # 只保留最新的N条数据
        self.pool = self.pool[-2_0000:]

    # 获取一批数据样本
    def sample(self):
        data = random.sample(self.pool, 64)

        state = (
            torch.FloatTensor(np.array([i[0] for i in data]))
            .reshape(-1, 3)
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
            .reshape(-1, 3)
            .to(self.device)
        )
        over = (
            torch.LongTensor(np.array([i[4] for i in data]))
            .reshape(-1, 1)
            .to(self.device)
        )

        return state, action, reward, next_state, over


if __name__ == "__main__":
    dqnModel = DQNModel()
    model_action = dqnModel.model_action

    player = Player(model_action)
    pool = Pool(player)

    pool.update()
    state, action, reward, next_state, over = pool.sample()
    print(state.shape, action.shape, reward.shape, next_state.shape, over.shape)
