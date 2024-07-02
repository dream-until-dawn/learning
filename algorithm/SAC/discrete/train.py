import torch
import torch.nn as nn

from model import DQNModel
from player import Player, Pool


class DQNLearning:
    def __init__(
        self,
        model_action: nn.Sequential,
        model_value1: nn.Sequential,
        model_value2: nn.Sequential,
        model_value1_next: nn.Sequential,
        model_value2_next: nn.Sequential,
        player: Player,
        pool: Pool,
    ):
        self.player = player
        self.pool = pool
        self.model_action = model_action
        self.model_value1 = model_value1
        self.model_value2 = model_value2
        self.model_value1_next = model_value1_next
        self.model_value2_next = model_value2_next
        self.optimizer_action = torch.optim.Adam(model_action.parameters(), lr=2e-4)
        self.optimizer_value1 = torch.optim.Adam(model_value1.parameters(), lr=2e-3)
        self.optimizer_value2 = torch.optim.Adam(model_value2.parameters(), lr=2e-3)
        self.alpha = 1.0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def soft_update(self, _from: nn.Sequential, _to: nn.Sequential):
        for _from, _to in zip(_from.parameters(), _to.parameters()):
            value = _to.data * 0.995 + _from.data * 0.005
            _to.data.copy_(value)

    def get_prob_entropy(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        s_tensor = state.float().reshape(-1, 4).to(self.device)
        prob: torch.Tensor = self.model_action(s_tensor)
        entropy: torch.Tensor = prob * (prob + 1e-8).log()
        entropy: torch.Tensor = -entropy.sum(dim=1, keepdim=True)

        return prob, entropy

    def requires_grad(self, model: nn.Sequential, value: bool) -> None:
        for param in model.parameters():
            param.requires_grad_(value)

    def train_value(self, state, action, reward, next_state, over):
        self.requires_grad(self.model_value1, True)
        self.requires_grad(self.model_value2, True)
        self.requires_grad(self.model_action, False)

        # 计算target
        with torch.no_grad():
            # 计算动作的熵
            prob, entropy = self.get_prob_entropy(next_state)
            target1: torch.Tensor = self.model_value1_next(next_state)
            target2: torch.Tensor = self.model_value2_next(next_state)
            target = torch.min(target1, target2)

        # 加权熵,熵越大越好
        target = (prob * target).sum(dim=1, keepdim=True)
        target = target + self.alpha * entropy
        target = target * 0.98 * (1 - over) + reward

        # 计算value
        value = self.model_value1(state).gather(dim=1, index=action)
        loss = torch.nn.functional.mse_loss(value, target)
        loss.backward()
        self.optimizer_value1.step()
        self.optimizer_value1.zero_grad()

        value = self.model_value2(state).gather(dim=1, index=action)
        loss = torch.nn.functional.mse_loss(value, target)
        loss.backward()
        self.optimizer_value2.step()
        self.optimizer_value2.zero_grad()

        return loss.item()

    def train_action(self, state):
        self.requires_grad(self.model_value1, False)
        self.requires_grad(self.model_value2, False)
        self.requires_grad(self.model_action, True)

        # 计算熵
        prob, entropy = self.get_prob_entropy(state)

        # 计算value
        value1 = self.model_value1(state)
        value2 = self.model_value2(state)
        value: torch.Tensor = torch.min(value1, value2)

        # 求期望求和
        value = (prob * value).sum(dim=1, keepdim=True)

        # 加权熵
        loss = -(value + self.alpha * entropy).mean()

        loss.backward()
        self.optimizer_action.step()
        self.optimizer_action.zero_grad()

        return loss.item()

    def train(self):
        # 训练N次
        for epoch in range(200):
            # 更新N条数据
            self.pool.update()

            # 每次更新过数据后,学习N次
            for i in range(200):
                # 采样一批数据
                state, action, reward, next_state, over = self.pool.sample()

                # 训练
                self.train_value(state, action, reward, next_state, over)
                self.train_action(state)
                self.soft_update(self.model_value1, self.model_value1_next)
                self.soft_update(self.model_value2, self.model_value2_next)

            self.alpha *= 0.9

            if epoch % 10 == 0:
                test_result = sum([self.player.play()[-1] for _ in range(20)]) / 20
                print(epoch, len(self.pool), self.alpha, test_result)


if __name__ == "__main__":
    if False:
        dqnModel = DQNModel()

        player = Player(dqnModel.model_action)
        pool = Pool(player)

        dqnLearning = DQNLearning(
            dqnModel.model_action,
            dqnModel.model_value1,
            dqnModel.model_value2,
            dqnModel.model_value1_next,
            dqnModel.model_value2_next,
            player,
            pool,
        )
        dqnLearning.train()
        dqnModel.saveModel(
            dqnLearning.model_action, dqnLearning.model_value1, dqnLearning.model_value2
        )
    else:
        dqnModel = DQNModel(mode="eval")
        dqnModel.loadModel()

        player = Player(dqnModel.model_action)
        player.play(show=True)
