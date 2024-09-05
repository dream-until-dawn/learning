import time

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from model import DQNModel
from player import Player
import config


class DQNLearning:
    def __init__(
        self,
        model: nn.Sequential,
        model_delay: nn.Sequential,
        player: Player,
        count: int = 10000,
    ):
        self.player = player
        self.model = model
        self.model_delay = model_delay
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.writer = SummaryWriter(
            log_dir=f"{config.log_dir}/{time.strftime('%Y-%m-%d-%H-%M')}"
        )
        self.count = count

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-4)
        loss_fn = torch.nn.MSELoss()
        revenue_list = []
        # 共更新N轮数据
        for epoch in range(self.count):
            # 更新模型参数
            self.player.update_model(self.model)
            # 采样数据
            (
                record_state,
                record_action,
                record_reward,
                record_next_state,
                record_over,
                reward_sum,
            ) = player.play()
            print(f"第{epoch+1:4d}\t轮,总奖励为\t{reward_sum:.4f}")
            self.writer.add_scalar("tarin/reward sum", reward_sum, epoch)  # 记录
            self.writer.add_scalar(
                "tarin/total revenue", self.player.env.total_revenue, epoch
            )  # 记录
            revenue_list.append(self.player.env.total_revenue)
            print(
                f"cur:{self.player.env.total_revenue:.2f}\tmin:{min(revenue_list):.2f}\tmax:{max(revenue_list):.2f}\tavg:{sum(revenue_list)/len(revenue_list):.2f}"
            )
            print("----------", "end", "----------")

            # 计算value
            value = self.model(record_state).gather(dim=1, index=record_action)

            # 计算target
            with torch.no_grad():
                # 使用原模型计算动作,使用延迟模型计算target,进一步缓解自举
                next_action = self.model(record_next_state).argmax(dim=1, keepdim=True)
                target = self.model_delay(record_next_state).gather(
                    dim=1, index=next_action
                )[0]
            target = target * 0.99 + record_reward

            loss = loss_fn(value, target)
            self.writer.add_scalar(
                "tarin/Policy Loss", loss.item(), epoch
            )  # 记录损失值
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # 复制参数
            if (epoch + 1) % 5 == 0:
                self.model_delay.load_state_dict(self.model.state_dict())


if __name__ == "__main__":
    if True:
        dqnModel = DQNModel()
        dqnModel.loadModel()
        player = Player(dqnModel.model)

        dqnLearning = DQNLearning(dqnModel.model, dqnModel.model_delay, player)
        dqnLearning.train()
        dqnModel.saveModel(dqnLearning.model)
    else:
        dqnModel = DQNModel(mode="eval")
        dqnModel.loadModel()
        player = Player(dqnModel.model)

        dqnLearning = DQNLearning(dqnModel.model, dqnModel.model_delay, player, 10)
        dqnLearning.train()
