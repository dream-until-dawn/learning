import time
import random

from IPython import display
import torch
import torch.nn as nn
import numpy as np

from env import MyWrapper
from model import DQNModel


class Player:
    def __init__(self, model: nn.Sequential):
        self.env = MyWrapper()
        self.model = model

    # 玩一局游戏并记录数据
    def play(self, show=False):
        data = []
        reward_sum = 0

        state = self.env.reset()
        over = False
        while not over:
            action = self.model(torch.FloatTensor(state).reshape(1, 4)).argmax().item()
            if random.random() < 0.1:
                action = self.env.action_space.sample()

            next_state, reward, over, _ = self.env.step(action)

            data.append((state, action, reward, next_state, over))
            reward_sum += reward

            state = next_state

            if show:
                # display.clear_output(wait=True)
                self.env.show()
                print(f"action: {action}\tReward: {reward}\tover: {over}")
                time.sleep(0.1)

        return data, reward_sum


# 数据池
class Pool:

    def __init__(self, player: Player):
        self.pool = []
        self.player = player

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

        state = torch.FloatTensor(np.array([i[0] for i in data])).reshape(-1, 4)
        action = torch.LongTensor(np.array([i[1] for i in data])).reshape(-1, 1)
        reward = torch.FloatTensor(np.array([i[2] for i in data])).reshape(-1, 1)
        next_state = torch.FloatTensor(np.array([i[3] for i in data])).reshape(-1, 4)
        over = torch.LongTensor(np.array([i[4] for i in data])).reshape(-1, 1)

        return state, action, reward, next_state, over


if __name__ == "__main__":
    dqnModel = DQNModel()
    model = dqnModel.model

    player = Player(model)

    pool = Pool(player)
    data, reward_sum = player.play()
    for row in data:
        print(row)
    print(reward_sum)
