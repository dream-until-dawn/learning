import torch

from model import DQNModel
from player import Pool
from player import Player


class DQNLearning:
    def __init__(
        self,
        model: DQNModel,
        model_delay: DQNModel,
        player: Player,
        pool: Pool,
    ):
        self.pool = pool
        self.player = player
        self.model = model
        self.model_delay = model_delay

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-4)
        loss_fn = torch.nn.MSELoss()

        # 共更新N轮数据
        for epoch in range(500):
            self.pool.update()

            # 每次更新数据后,训练N次
            for i in range(200):

                # 采样N条数据
                state, action, reward, next_state, over = self.pool.sample()

                # 计算value
                value = self.model(state).gather(dim=1, index=action)

                # 计算target
                with torch.no_grad():
                    target = self.model_delay(next_state)
                target = target.max(dim=1)[0].reshape(-1, 1)
                target = target * 0.99 * (1 - over) + reward

                loss = loss_fn(value, target)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            # 复制参数
            if (epoch + 1) % 5 == 0:
                self.model_delay.load_state_dict(self.model.state_dict())

            if epoch % 100 == 0:
                test_result = sum([self.player.play()[-1] for _ in range(20)]) / 20
                print(epoch, len(self.pool), test_result)


if __name__ == "__main__":
    if False:
        model = DQNModel()
        model_delay = DQNModel()
        model_delay.load_state_dict(model.state_dict())

        player = Player(model)

        pool = Pool(player)

        dqnLearning = DQNLearning(model, model_delay, player, pool)
        dqnLearning.train()
        model.saveModel(dqnLearning.model)
    else:
        dqnModel = DQNModel(mode="eval")
        dqnModel.loadModel()

        player = Player(dqnModel)
        player.play(show=True)
