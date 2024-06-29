import torch
import torch.nn as nn

from model import DQNModel
from player import Player


class DQNLearning:
    def __init__(
        self,
        model_actor: nn.Sequential,
        model_critic: nn.Sequential,
        model_critic_delay: nn.Sequential,
        player: Player,
    ):
        self.player = player
        self.model_actor = model_actor
        self.model_critic = model_critic
        self.model_critic_delay = model_critic_delay
        self.optimizer_actor = torch.optim.Adam(self.model_actor.parameters(), lr=4e-3)
        self.optimizer_critic = torch.optim.Adam(
            self.model_critic.parameters(), lr=4e-2
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def requires_grad(self, model: nn.Sequential, value: bool) -> None:
        for param in model.parameters():
            param.requires_grad_(value)

    def train_critic(self, state, reward, next_state, over):
        self.requires_grad(self.model_actor, False)
        self.requires_grad(self.model_critic, True)

        # 计算values和targets
        value: torch.Tensor = self.model_critic(state)

        with torch.no_grad():
            target = self.model_critic_delay(next_state)
        target = target * 0.99 * (1 - over) + reward

        # 时序差分误差,也就是tdloss
        loss = torch.nn.functional.mse_loss(value, target)

        loss.backward()
        self.optimizer_critic.step()
        self.optimizer_critic.zero_grad()

        return value.detach()

    def train_actor(self, state, action, value):
        self.requires_grad(self.model_actor, True)
        self.requires_grad(self.model_critic, False)

        # 重新计算动作的概率
        prob: torch.Tensor = self.model_actor(state)
        prob = prob.gather(dim=1, index=action)

        # 根据策略梯度算法的导函数实现
        # 函数中的Q(state,action),这里使用critic模型估算
        prob = (prob + 1e-8).log() * value
        loss = -prob.mean()

        loss.backward()
        self.optimizer_actor.step()
        self.optimizer_actor.zero_grad()

        return loss.item()

    def train(self):
        # 训练N局
        for epoch in range(500):

            # 一个epoch最少玩N步
            steps = 0
            while steps < 200:
                state, action, reward, next_state, over, _ = self.player.play()
                steps += len(state)

                # 训练两个模型
                value = self.train_critic(state, reward, next_state, over)
                loss = self.train_actor(state, action, value)

            # 复制参数
            for param, param_delay in zip(
                self.model_critic.parameters(), self.model_critic_delay.parameters()
            ):
                value = param_delay.data * 0.7 + param.data * 0.3
                param_delay.data.copy_(value)

            if epoch % 100 == 0:
                test_result = sum([self.player.play()[-1] for _ in range(20)]) / 20
                print(epoch, loss, test_result)


if __name__ == "__main__":
    if False:
        dqnModel = DQNModel()

        player = Player(dqnModel.model_actor)

        dqnLearning = DQNLearning(
            dqnModel.model_actor,
            dqnModel.model_critic,
            dqnModel.model_critic_delay,
            player,
        )
        dqnLearning.train()
        dqnModel.saveModel(dqnLearning.model_actor, dqnLearning.model_critic)
    else:
        dqnModel = DQNModel(mode="eval")
        dqnModel.loadModel()

        player = Player(dqnModel.model_actor)
        player.play(show=True)
