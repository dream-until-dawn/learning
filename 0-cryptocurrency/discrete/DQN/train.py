import time

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from model import DQNModel
from player import Player
import config


class DQNLearning:
    def __init__(self, model: nn.Sequential, player: Player):
        self.player = player
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.writer = SummaryWriter(
            log_dir=f"{config.log_dir}/{time.strftime('%Y-%m-%d-%H-%M')}"
        )

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        loss_fn = torch.nn.MSELoss()

        # 共更新N轮数据
        for epoch in range(500):
            # 更新模型参数
            self.player.updata_model(self.model)
            # 采样数据
            (
                record_state,
                record_action,
                record_reward,
                record_next_state,
                record_over,
                reward_sum,
            ) = player.play()
            print(epoch, reward_sum)
            self.writer.add_scalar("tarin/reward sum", reward_sum, epoch)  # 记录
            self.writer.add_scalar(
                "tarin/total revenue", self.player.env.total_revenue, epoch
            )  # 记录
            # 计算value
            value = self.model(record_state).gather(dim=1, index=record_action)

            # 计算target
            with torch.no_grad():
                target = self.model(record_next_state)
            target = target.max(dim=1, keepdim=True)[0].reshape(-1, 1)
            target = target * 0.99 + record_reward

            loss = loss_fn(value, target)
            self.writer.add_scalar(
                "tarin/Policy Loss", loss.item(), epoch
            )  # 记录损失值
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


if __name__ == "__main__":
    if True:
        dqnModel = DQNModel()
        dqnModel.loadModel()
        player = Player(dqnModel.model)

        dqnLearning = DQNLearning(dqnModel.model, player)
        dqnLearning.train()
        dqnModel.saveModel(dqnLearning.model)
    else:
        dqnModel = DQNModel(mode="eval")
        dqnModel.loadModel()

        player = Player(dqnModel.model)
        player.play(show=True)
