import random

import torch
import torch.nn as nn
import numpy as np

from env import MyWrapper
from model import DQNModel


class Player:
    def __init__(self, model: nn.Sequential):
        self.env = MyWrapper()
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def updata_model(self, model: nn.Sequential):
        self.model = model

    def check_nan(self, tensor, name):
        nan_indices = torch.where(torch.isnan(tensor))
        inf_indices = torch.where(torch.isinf(tensor))

        if nan_indices[0].numel() > 0:
            print(f"{name} contains NaN at positions: {nan_indices}")

        if inf_indices[0].numel() > 0:
            print(f"{name} contains Inf at positions: {inf_indices}")

    # 玩一局游戏并记录数据
    def play(self):
        record_state = []
        record_action = []
        record_reward = []
        record_next_state = []
        record_over = []
        reward_sum = 0

        state = self.env.reset()
        over = False
        while not over:
            action = (
                self.model(torch.FloatTensor(state).to(self.device)).argmax().item()
            )
            # 随机动作概率为0.1
            if random.random() < 0.1:
                action = self.env.random_action()

            next_state, reward, over, _ = self.env.step(action)
            record_state.append(state)
            record_action.append(action)
            record_reward.append(reward)
            record_next_state.append(next_state)
            record_over.append(over)

            reward_sum += reward

            state = next_state

        record_state = torch.FloatTensor(np.array(record_state)).to(self.device)
        record_action = (
            torch.LongTensor(np.array(record_action)).reshape(-1, 1).to(self.device)
        )
        record_reward = (
            torch.FloatTensor(np.array(record_reward)).reshape(-1, 1).to(self.device)
        )
        record_next_state = torch.FloatTensor(np.array(record_next_state)).to(
            self.device
        )
        record_over = torch.FloatTensor(np.array(record_over)).to(self.device)

        self.check_nan(record_state, "record_state")
        self.check_nan(record_next_state, "record_next_state")
        return (
            record_state,
            record_action,
            record_reward,
            record_next_state,
            record_over,
            reward_sum,
        )


if __name__ == "__main__":
    dqnModel = DQNModel()
    dqnModel.loadModel()
    player = Player(dqnModel.model)
    (
        record_state,
        record_action,
        record_reward,
        record_next_state,
        record_over,
        reward_sum,
    ) = player.play()
    print(record_state.shape)
    print(record_action.shape)
    print(record_reward.shape)
    print(record_next_state.shape)
    print(reward_sum)
