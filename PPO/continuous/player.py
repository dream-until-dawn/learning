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
        state = []
        action = []
        reward = []
        next_state = []
        over = []

        s = self.env.reset()
        o = False
        while not o:
            # 根据概率采样
            s_tensor = torch.FloatTensor(s).reshape(1, 3).to(self.device)
            mu, sigma = self.model_action(s_tensor)
            a = [random.normalvariate(mu=mu.item(), sigma=sigma.item())]
            ns, r, o, _ = self.env.step(a)

            if show:
                display.clear_output(wait=True)
                self.env.show()
                print(f"state: {s}, action: {a}, reward: {r}")
            else:
                state.append(s)
                action.append(a)
                reward.append(r)
                next_state.append(ns)
                over.append(o)
            s = ns

        state = torch.FloatTensor(np.array(state)).reshape(-1, 3).to(self.device)
        action = torch.FloatTensor(np.array(action)).reshape(-1, 1).to(self.device)
        reward = torch.FloatTensor(np.array(reward)).reshape(-1, 1).to(self.device)
        next_state = (
            torch.FloatTensor(np.array(next_state)).reshape(-1, 3).to(self.device)
        )
        over = torch.LongTensor(np.array(over)).reshape(-1, 1).to(self.device)

        return state, action, reward, next_state, over, reward.sum().item()


if __name__ == "__main__":
    dqnModel = DQNModel()
    model_action = dqnModel.model_action

    player = Player(model_action)

    state, action, reward, next_state, over, _ = player.play(True)
    for i in range(len(state)):
        print(f"state: {state[i]}, action: {action[i]}, reward: {reward[i]}")
