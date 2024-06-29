import torch
import torch.nn as nn

from model import DQNModel
from player import Player


class DQNLearning:
    def __init__(
        self,
        model_action: nn.Sequential,
        model_value: nn.Sequential,
        player: Player,
    ):
        self.player = player
        self.model_action = model_action
        self.model_value = model_value
        self.optimizer_action = torch.optim.Adam(model_action.parameters(), lr=5e-4)
        self.optimizer_value = torch.optim.Adam(model_value.parameters(), lr=5e-3)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def requires_grad(self, model: nn.Sequential, value: bool) -> None:
        for param in model.parameters():
            param.requires_grad_(value)

    def train_value(self, state, reward, next_state, over):
        self.requires_grad(self.model_action, False)
        self.requires_grad(self.model_value, True)

        # 计算target
        with torch.no_grad():
            target: torch.Tensor = self.model_value(next_state)
        target: torch.Tensor = target * 0.98 * (1 - over) + reward

        # 每批数据反复训练10次
        for _ in range(10):
            # 计算value
            value = self.model_value(state)

            loss = torch.nn.functional.mse_loss(value, target)
            loss.backward()
            self.optimizer_value.step()
            self.optimizer_value.zero_grad()

        # 减去value相当于去基线
        return (target - value).detach()

    def train_action(self, state, action, value):
        self.requires_grad(self.model_action, True)
        self.requires_grad(self.model_value, False)

        # 计算当前state的价值,其实就是Q(state,action),这里是用蒙特卡洛法估计的
        delta = []
        for i in range(len(value)):
            s = 0
            for j in range(i, len(value)):
                s += value[j] * (0.9 * 0.9) ** (j - i)
            delta.append(s)
        delta = torch.FloatTensor(delta).reshape(-1, 1).to(self.device)

        # 更新前的动作概率
        with torch.no_grad():
            mu, sigma = self.model_action(state)
            prob_old = torch.distributions.Normal(mu, sigma).log_prob(action).exp()

        # 每批数据反复训练10次
        for _ in range(10):
            # 更新后的动作概率
            mu, sigma = self.model_action(state)
            prob_new = torch.distributions.Normal(mu, sigma).log_prob(action).exp()

            # 求出概率的变化
            ratio: torch.Tensor = prob_new / prob_old

            # 计算截断的和不截断的两份loss,取其中小的
            surr1 = ratio * delta
            surr2 = ratio.clamp(0.8, 1.2) * delta

            loss = -torch.min(surr1, surr2).mean()

            # 更新参数
            loss.backward()
            self.optimizer_action.step()
            self.optimizer_action.zero_grad()

        return loss.item()

    def train(self):
        # 训练N局
        for epoch in range(1000):
            # 一个epoch最少玩N步
            steps = 0
            while steps < 200:
                state, action, reward, next_state, over, _ = self.player.play()
                steps += len(state)

                # 训练两个模型
                delta = self.train_value(state, reward, next_state, over)
                loss = self.train_action(state, action, delta)

            if epoch % 100 == 0:
                test_result = sum([self.player.play()[-1] for _ in range(20)]) / 20
                print(epoch, loss, test_result)


if __name__ == "__main__":
    if False:
        dqnModel = DQNModel()

        player = Player(dqnModel.model_action)

        dqnLearning = DQNLearning(
            dqnModel.model_action,
            dqnModel.model_value,
            player,
        )
        dqnLearning.train()
        dqnModel.saveModel(dqnLearning.model_action, dqnLearning.model_value)
    else:
        dqnModel = DQNModel(mode="eval")
        dqnModel.loadModel()

        player = Player(dqnModel.model_action)
        player.play(show=True)
