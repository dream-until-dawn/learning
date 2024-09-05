import random

import torch
import torch.nn as nn
import numpy as np

from env import MyENV
from model import DQNModel


class Player:
    def __init__(self, model: nn.Sequential, is_train=True):
        self.env = MyENV()
        self.model = model
        self.is_train = is_train
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dict = {0: "hlod", 1: "O L", 2: "O S", 3: "C L", 4: "C S"}

    def update_model(self, model: nn.Sequential):
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
                self.model(torch.FloatTensor(state.reshape(-1)).to(self.device))
                .argmax()
                .item()
            )
            # 随机动作概率为0.05
            if self.is_train and random.random() < 0.05:
                action = self.env.random_action()

            next_state, reward, over, _ = self.env.step(action)
            record_state.append(state.reshape(-1))
            record_action.append(action)
            record_reward.append(reward)
            record_next_state.append(next_state.reshape(-1))
            record_over.append(over)

            reward_sum += reward

            state = next_state

        record_reward = np.array(record_reward)
        count = np.count_nonzero(record_reward == -99)
        if count > 0:
            print(f"动作错误次数: {count}")
        record_action = np.array(record_action)
        unique_action, action_counts = np.unique(record_action, return_counts=True)
        # 计算每个元素出现的概率
        probabilities = action_counts / len(record_action)
        print(f"total: {len(record_action)}", end="\t")
        for element, prob in zip(unique_action, probabilities):
            print(f"{self.action_dict[element]}: {prob*100:.2f}%", end="\t")
        print(
            f"\nmax reward: {np.max(record_reward):.4f}\tmin reward: {np.min(record_reward[record_reward != -99]):.4f}"
        )
        record_state = torch.FloatTensor(np.array(record_state)).to(self.device)
        record_action = torch.LongTensor(record_action).reshape(-1, 1).to(self.device)
        record_reward = torch.FloatTensor(record_reward).reshape(-1, 1).to(self.device)
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
