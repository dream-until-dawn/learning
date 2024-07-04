import random

import torch
from torch.distributions import Normal
import numpy as np

from env import MyWrapper
from model import PPOModel, MyModel
import config


class Player:
    def __init__(self, model_action: PPOModel):
        self.env = MyWrapper()
        self.model_action = model_action
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 玩一局游戏并记录数据
    def play(self):
        state = []
        action = []
        reward = []
        next_state = []
        over = []

        s = self.env.reset()
        o = False
        while not o:
            # 根据概率采样
            s_tensor = torch.FloatTensor(s).reshape(1, config.prev_dim).to(self.device)
            mu, sigma = self.model_action(s_tensor)
            mu_value = mu.item()
            sigma_value = sigma.item()
            # 创建正态分布
            normal_dist = Normal(mu_value, sigma_value)
            # 生成服从正态分布的随机数，并截断在0到1之间
            a = normal_dist.sample().clamp(0, 1).item()
            # 10%的概率随机动作
            if random.uniform(0, 1) <= 0.1:
                a = random.uniform(0, 1)

            ns, r, o, _ = self.env.step(a)

            state.append(s)
            s = ns
            action.append(a)
            reward.append(r)
            next_state.append(ns)
            over.append(o)

        state = (
            torch.FloatTensor(np.array(state))
            .reshape(-1, config.prev_dim)
            .to(self.device)
        )
        action = torch.FloatTensor(np.array(action)).reshape(-1, 1).to(self.device)
        reward = torch.FloatTensor(np.array(reward)).reshape(-1, 1).to(self.device)
        next_state = (
            torch.FloatTensor(np.array(next_state))
            .reshape(-1, config.prev_dim)
            .to(self.device)
        )
        over = torch.LongTensor(np.array(over)).reshape(-1, 1).to(self.device)

        return state, action, reward, next_state, over, reward.sum().item()


if __name__ == "__main__":
    myModel = MyModel()
    myModel.loadModel()
    player = Player(myModel.model_action)

    for _ in range(20):
        state, action, reward, next_state, over, _ = player.play()
        print(state.shape, action.shape, reward.shape, next_state.shape, over.shape)
