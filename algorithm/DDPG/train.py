import torch
import torch.nn as nn

from model import DQNModel
from player import Player, Pool


class DQNLearning:
    def __init__(
        self,
        model_action: nn.Sequential,
        model_action_delay: nn.Sequential,
        model_value: nn.Sequential,
        model_value_delay: nn.Sequential,
        player: Player,
        pool: Pool,
    ):
        self.player = player
        self.pool = pool
        self.model_action = model_action
        self.model_action_delay = model_action_delay
        self.model_value = model_value
        self.model_value_delay = model_value_delay
        self.optimizer_action = torch.optim.Adam(
            self.model_action.parameters(), lr=5e-4
        )
        self.optimizer_value = torch.optim.Adam(self.model_value.parameters(), lr=5e-3)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.requires_grad(self.model_action_delay, False)
        self.requires_grad(self.model_value_delay, False)

    def soft_update(self, _from: nn.Sequential, _to: nn.Sequential) -> None:
        for _from, _to in zip(_from.parameters(), _to.parameters()):
            value = _to.data * 0.7 + _from.data * 0.3
            _to.data.copy_(value)

    def requires_grad(self, model: nn.Sequential, value: bool) -> None:
        for param in model.parameters():
            param.requires_grad_(value)

    def train_action(self, state):
        self.requires_grad(self.model_action, True)
        self.requires_grad(self.model_value, False)

        # 首先把动作计算出来
        action = self.model_action(state)

        # 使用value网络评估动作的价值,价值是越高越好
        input = torch.cat([state, action], dim=1)
        loss = -self.model_value(input).mean()

        loss.backward()
        self.optimizer_action.step()
        self.optimizer_action.zero_grad()

        return loss.item()

    def train_value(self, state, action, reward, next_state, over):
        self.requires_grad(self.model_action, False)
        self.requires_grad(self.model_value, True)

        # 计算value
        input = torch.cat([state, action], dim=1)
        value = self.model_value(input)

        # 计算target
        with torch.no_grad():
            next_action = self.model_action_delay(next_state)
            input = torch.cat([next_state, next_action], dim=1)
            target = self.model_value_delay(input)
        target = target * 0.99 * (1 - over) + reward

        # 计算td loss,更新参数
        loss = torch.nn.functional.mse_loss(value, target)

        loss.backward()
        self.optimizer_value.step()
        self.optimizer_value.zero_grad()

        return loss.item()

    # 训练
    def train(self):

        # 共更新N轮数据
        for epoch in range(200):
            self.pool.update()

            # 每次更新数据后,训练N次
            for i in range(200):

                # 采样N条数据
                state, action, reward, next_state, over = self.pool.sample()

                # 训练模型
                self.train_action(state)
                self.train_value(state, action, reward, next_state, over)

            self.soft_update(self.model_action, self.model_action_delay)
            self.soft_update(self.model_value, self.model_value_delay)

            if epoch % 20 == 0:
                test_result = sum([self.player.play()[-1] for _ in range(20)]) / 20
                print(epoch, len(self.pool), test_result)


if __name__ == "__main__":
    if False:
        dqnModel = DQNModel()

        player = Player(dqnModel.model_action)
        pool = Pool(player)

        dqnLearning = DQNLearning(
            dqnModel.model_action,
            dqnModel.model_action_delay,
            dqnModel.model_value,
            dqnModel.model_value_delay,
            player,
            pool,
        )
        dqnLearning.train()
        dqnModel.saveModel(
            dqnLearning.model_action,
            dqnLearning.model_value,
        )
    else:
        dqnModel = DQNModel(mode="eval")
        dqnModel.loadModel()

        player = Player(dqnModel.model_action)
        player.play(show=True)
