import random

from IPython import display
import numpy as np

from env import MyWrapper


class Player:
    def __init__(self, env: MyWrapper):
        self.env = env
        self.Q = np.zeros((16, 4))

    # 玩一局游戏并记录数据
    def play(self, show=False):
        data = []
        reward_sum = 0
        state = self.env.reset()
        over = False
        while not over:
            action = self.Q[state].argmax()
            if random.random() < 0.1:
                action = self.env.action_space.sample()

            next_state, reward, over, _ = self.env.step(action)

            data.append((state, action, reward, next_state, over))
            reward_sum += reward

            state = next_state

            if show:
                display.clear_output(wait=True)
                self.env.show()

        return data, reward_sum


# 数据池
class Pool:

    def __init__(self, player: Player):
        self.pool = []
        self.player = player

    def __len__(self):
        return len(self.pool)

    def __getitem__(self, i):
        return self.pool[i]

    # 更新动作池
    def update(self):
        # 每次更新不少于N条新数据
        old_len = len(self.pool)
        while len(self.pool) - old_len < 200:
            self.pool.extend(self.player.play()[0])

        # 只保留最新的N条数据
        self.pool = self.pool[-1_0000:]

    # 获取一批数据样本
    def sample(self):
        return random.choice(self.pool)


if __name__ == "__main__":
    env = MyWrapper()
    player = Player(env)
    pool = Pool(player)
    data, reward_sum = player.play()
    print(data)
    print(reward_sum)
