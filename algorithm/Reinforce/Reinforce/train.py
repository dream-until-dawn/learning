import torch
import torch.nn as nn

from model import DQNModel
from player import Player


class DQNLearning:
    def __init__(self, model: nn.Sequential, player: Player):
        self.player = player
        self.model = model

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-3)

        # 训练N局
        for epoch in range(500):

            # 一个epoch最少玩N步
            steps = 0
            while steps < 200:

                # 玩一局游戏,得到数据
                state, action, reward, _ = self.player.play()
                steps += len(state)

                # 计算当前state的价值,其实就是Q(state,action),这里是用蒙特卡洛法估计的
                value = []
                for i in range(len(reward)):
                    s = 0
                    for j in range(i, len(reward)):
                        s += reward[j] * 0.99 ** (j - i)
                    value.append(s)
                value = torch.FloatTensor(value).reshape(-1, 1)

                # 重新计算动作的概率
                prob = self.model(state).gather(dim=1, index=action)

                # 求Q最大的导函数 -> partial value / partial action
                prob = (prob + 1e-8).log() * value
                for i in range(len(prob)):
                    prob[i] = prob[i] * 0.99**i
                loss = -prob.mean()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            if epoch % 100 == 0:
                test_result = sum([self.player.play()[-1] for _ in range(20)]) / 20
                print(epoch, loss.item(), test_result)


if __name__ == "__main__":
    if False:
        dqnModel = DQNModel()

        player = Player(dqnModel.model)

        dqnLearning = DQNLearning(dqnModel.model, player)
        dqnLearning.train()
        dqnModel.saveModel(dqnLearning.model)
    else:
        dqnModel = DQNModel(mode="eval")
        dqnModel.loadModel()

        player = Player(dqnModel.model)
        player.play(show=True)
