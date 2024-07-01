import random
import torch
import torch.nn as nn
import numpy as np


from env import MyWrapper
from model import LearningModel, PPOModel


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
            s_tensor = torch.FloatTensor(s).reshape(1, 88).to(self.device)
            a: torch.Tensor = self.model_action(s_tensor)
            ns, r, o, _ = self.env.step(a)

            s = ns
            state.append(s)
            action.append(a.detach().numpy()[0])
            reward.append(r.detach().numpy()[0])
            next_state.append(ns)
            over.append(o)
        state = torch.FloatTensor(np.array(state)).reshape(-1, 88).to(self.device)
        action = torch.FloatTensor(np.array(action)).reshape(-1, 1).to(self.device)
        reward = torch.FloatTensor(np.array(reward)).reshape(-1, 1).to(self.device)
        next_state = (
            torch.FloatTensor(np.array(next_state)).reshape(-1, 88).to(self.device)
        )
        over = torch.LongTensor(np.array(over)).reshape(-1, 1).to(self.device)

        return state, action, reward, next_state, over, reward.sum().item()


if __name__ == "__main__":
    learningModel = LearningModel()
    print(learningModel.model_action)
    player = Player(learningModel.model_action)

    state, action, reward, next_state, over, reward_sum = player.play()
    print(f"state: {state.shape}")
    print(f"action: {action.shape}")
    print(f"reward: {reward.shape}")
    print(f"next_state: {next_state.shape}")
    print(f"reward_sum: {reward_sum}")
    # for i in range(len(state)):
    #     print(f"state: {state[i]}, action: {action[i]}, reward: {reward[i]}")
