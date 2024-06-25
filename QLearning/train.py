from player import Pool


class QLearning:
    def __init__(self):
        self.pool = Pool()

    def train(self):
        # 共更新N轮数据
        for epoch in range(1000):
            self.pool.update()

            # 每次更新数据后,训练N次
            for i in range(200):

                # 随机抽一条数据
                state, action, reward, next_state, _ = self.pool.sample()

                # Q矩阵当前估计的state下action的价值
                value = self.pool.player.Q[state, action]

                # 实际玩了之后得到的reward+下一个状态的价值*0.9
                target = reward + self.pool.player.Q[next_state].max() * 0.9

                # value和target应该是相等的,说明Q矩阵的评估准确
                # 如果有误差,则应该以target为准更新Q表,修正它的偏差
                # 这就是TD误差,指评估值之间的偏差,以实际成分高的评估为准进行修正
                update = (target - value) * 0.1

                # 更新Q表
                self.pool.player.Q[state, action] += update

            if epoch % 100 == 0:
                print(epoch, len(self.pool), self.pool.player.play()[-1])
                print(self.pool.player.Q)


if __name__ == "__main__":
    qLearning = QLearning()
    qLearning.train()
    # qLearning.pool.player.play(show=True)
