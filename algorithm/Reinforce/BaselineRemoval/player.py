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
        state = []
        action = []
        reward = []

        s = self.env.reset()
        o = False
        while not o:
            # 根据概率采样
            prob = self.model(torch.FloatTensor(s).reshape(1, 4))[0].tolist()
            a = random.choices(range(2), weights=prob, k=1)[0]

            ns, r, o, _ = self.env.step(a)

            state.append(s)
            action.append(a)
            reward.append(r)

            s = ns

            if show:
                display.clear_output(wait=True)
                self.env.show()

        state = torch.FloatTensor(np.array(state)).reshape(-1, 4)
        action = torch.LongTensor(np.array(action)).reshape(-1, 1)
        reward = torch.FloatTensor(np.array(reward)).reshape(-1, 1)

        return state, action, reward, reward.sum().item()


if __name__ == "__main__":
    dqnModel = DQNModel()
    model = dqnModel.model

    player = Player(model)

    data, reward_sum = player.play()
    for row in data:
        print(row)
    print(reward_sum)
