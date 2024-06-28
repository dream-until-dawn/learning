import torch
import torch.nn as nn

from model import DQNModel
from player import Player


class DQNLearning:
    def __init__(
        self,
        model: nn.Sequential,
        model_baseline: nn.Sequential,
        player: Player,
    ):
        self.player = player
        self.model_action = model
        self.model_baseline = model_baseline
        self.optimizer_action = torch.optim.Adam(
            self.model_action.parameters(), lr=5e-3
        )
        self.optimizer_baseline = torch.optim.Adam(
            self.model_baseline.parameters(), lr=5e-4
        )

    # 计算当前state的价值,其实就是Q(state,action),这里是用蒙特卡洛法估计的
    def get_value(self,reward):
        value = []
        for i in range(len(reward)):
            s = 0
            for j in range(i, len(reward)):
                s += reward[j] * 0.99 ** (j - i)
            value.append(s)
        return torch.FloatTensor(value).reshape(-1, 1)

    # 训练baseline模型
    def train_baseline(self, state, value):
        baseline = self.model_baseline(state)

        loss = torch.nn.functional.mse_loss(baseline, value)
        loss.backward()
        self.optimizer_baseline.step()
        self.optimizer_baseline.zero_grad()

        return baseline.detach()

    # 重新计算动作的概率
    def train_action(self, state, action, value, baseline):
        prob = self.model_action(state).gather(dim=1, index=action)

        # 求Q最大的导函数 -> partial value / partial action
        # 注意这里的Q使用前要去基线,这也是baseline模型存在的意义
        prob = (prob + 1e-8).log() * (value - baseline)
        for i in range(len(prob)):
            prob[i] = prob[i] * 0.99**i
        loss = -prob.mean()

        loss.backward()
        self.optimizer_action.step()
        self.optimizer_action.zero_grad()

        return loss.item()

    def train(self):
        self.model_action.train()
        self.model_baseline.train()

        # 训练N局
        for epoch in range(500):

            # 一个epoch最少玩N步
            steps = 0
            while steps < 200:

                # 玩一局游戏,得到数据
                state, action, reward, _ = self.player.play()
                steps += len(state)

                # 训练两个模型
                value = self.get_value(reward)
                baseline = self.train_baseline(state, value)
                loss = self.train_action(state, action, value, baseline)

            if epoch % 100 == 0:
                test_result = sum([self.player.play()[-1] for _ in range(20)]) / 20
                print(epoch, loss, test_result)


if __name__ == "__main__":
    if False:
        dqnModel = DQNModel()

        player = Player(dqnModel.model)

        dqnLearning = DQNLearning(dqnModel.model, dqnModel.model_baseline, player)
        dqnLearning.train()
        dqnModel.saveModel(dqnLearning.model_action, dqnModel.model_baseline)
    else:
        dqnModel = DQNModel(mode="eval")
        dqnModel.loadModel()

        player = Player(dqnModel.model)
        player.play(show=True)
