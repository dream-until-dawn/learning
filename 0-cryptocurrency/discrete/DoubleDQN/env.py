import gym
import pandas as pd
import numpy as np
import random
import time
from config import TECHNICAL_INDICATORS_LIST


class readDS:
    def __init__(self, open_file_name):
        self.open_file_name = f"DS/{open_file_name}"

    def pull_data(self, target="train") -> pd.DataFrame:
        return pd.read_csv(self.open_file_name + f"_{target}.csv")


# 定义环境
class MyENV(gym.Wrapper):

    def __init__(self, open_file_name="ETH-USDT-SWAP-1D", target="train"):
        self.read_ds = readDS(open_file_name)
        self.df = self.read_ds.pull_data(target)
        self.df = self.df[TECHNICAL_INDICATORS_LIST]  # 仅保留有使用到的列
        self.df = self.df.set_index("ts")  # 设置ts为索引
        self.state_col = ["dif", "dea", "macd"]  # 状态列
        self.episode = 0  # 回合数
        self.data_index = 0  # 当前索引
        self.total_steps = len(self.df)  # 总可用步数
        self.starting_point = 4  # 最小起始点
        self.ending_point = self.total_steps - 4  # 最大结束点
        self.play_steps = 90  # 3months
        print(f"df shape: {self.df.shape}")
        self.action_dict = {
            0: "hold",
            1: "#-long",
            2: "&-short",
            3: "#-c l",
            4: "&-c s",
        }
        self.PS = 0  # 仓位状态:0-未持仓，1-多头持仓，-1-空头持仓
        self.UG = 0  # 未实现收益
        self.total_revenue = 1  # 总奖励
        self.max_total_revenue = 1  # 最大总奖励
        self.print_head = True  # 打印头部信息

    @property  # 当前状态
    def current_state(self) -> pd.DataFrame:
        return self.df.iloc[self.data_index - 1 : self.data_index + 1][self.state_col]

    @property  # 下一状态
    def next_state(self) -> pd.DataFrame:
        return self.df.iloc[self.data_index : self.data_index + 2][self.state_col]

    @property  # 当前价格
    def current_price(self) -> np.float64:
        return self.df.iloc[self.data_index]["close"]

    @property  # 下一价格
    def next_price(self) -> np.float64:
        return self.df.iloc[self.data_index + 1]["close"]

    @property  # 当前日期
    def current_date(self) -> str:
        return self.df.iloc[self.data_index].name

    @property  # 明日涨幅
    def next_increase(self) -> np.float64:
        return self.round4((self.next_price - self.current_price) / self.current_price)

    def round4(self, x: float) -> float:
        return round(x, 4)

    # 当前状态
    def get_current_state(self) -> np.ndarray:
        a1 = [self.PS] * len(self.state_col)
        a2 = self.current_state.to_numpy()
        return np.vstack((a1, a2))

    # 下一状态
    def get_next_state(self) -> np.ndarray:
        a1 = [self.PS] * len(self.state_col)
        a2 = self.next_state.to_numpy()
        return np.vstack((a1, a2))

    # 设置随机种子
    def seed(self, seed=None) -> None:
        if seed is None:
            seed = int(round(time.time() * 1000))
        random.seed(seed)

    # 打印当前状态
    def check_print(self, action=0) -> None:
        if self.print_head:
            self.print_head = False  # 打印头部信息
            print(
                f"{'date':<20}{'action':<10}{'PS':<5}{'UG':<10}{'c_price':<10}{'next_increase':<10}"
            )
        print_ug = self.round4(self.UG)
        print(
            f"{self.current_date:<20}{self.action_dict[action]:<10}{self.PS:<5}{print_ug:<10}{self.current_price:<10}{self.next_increase:<10}"
        )

    # 重置环境
    def reset(self) -> np.ndarray:
        self.episode += 1
        self.seed()
        self.starting_point = random.choice(
            range(4, self.total_steps - self.play_steps - 4)
        )
        self.ending_point = self.starting_point + self.play_steps
        self.data_index = self.starting_point
        self.PS = 0
        self.UG = 0
        self.total_revenue = 1
        self.max_total_revenue = 1
        self.print_head = True  # 打印头部信息
        return self.get_current_state()

    # 执行动作
    def step(self, action) -> tuple[np.ndarray, float, bool, dict]:
        over = False
        next_state = self.get_next_state()  # 下一状态
        # 资金不足
        if self.total_revenue < 0.1:
            over = True
            print(f"error: 资金不足\t{self.total_revenue}")
            return next_state, -99, over, {}

        reward = 0
        # self.check_print(action)  # 打印当前状态
        if self.PS == 0:
            reward = self.noPosition(action)
        elif self.PS == 1:
            reward = self.longPosition(action)
        else:
            reward = self.shortPosition(action)

        self.data_index += 1
        # 是否已结束-超过限制 or 超过总数目
        if (
            self.data_index > self.ending_point
            or self.data_index > self.total_steps - 4
        ):
            over = True
            # print(f"end:步数已达到\t{self.total_revenue}")
            return next_state, reward, over, {}
        else:
            return next_state, reward, over, {}

    # 未持仓状态
    def noPosition(self, action) -> float:
        next_increase = self.next_increase
        abs_next_increase = abs(next_increase)
        if action == 0:
            if abs_next_increase > 0.01:
                return -abs_next_increase * 10
            else:
                return 0.05
        elif action == 1:
            self.PS = 1
            self.UG = next_increase - 0.0005
            return next_increase * 10
        elif action == 2:
            self.PS = -1
            self.UG = -next_increase - 0.0005
            return next_increase * 10
        else:
            return -99

    # 多头持仓状态
    def longPosition(self, action) -> float:
        next_increase = self.next_increase
        if action == 0:
            self.UG = self.round4((1 + self.UG) * (1 + next_increase) - 1)
            return next_increase * 10
        elif action == 3:
            self.total_revenue = self.round4(self.total_revenue * (1 + self.UG))
            self.max_total_revenue = max(self.max_total_revenue, self.total_revenue)
            self.PS = 0
            self.UG = 0
            return next_increase * 10
        else:
            return -99

    # 空头持仓状态
    def shortPosition(self, action) -> float:
        next_increase = -self.next_increase
        if action == 0:
            self.UG = self.round4((1 + self.UG) * (1 + next_increase) - 1)
            return next_increase * 10
        elif action == 4:
            self.total_revenue = self.round4(self.total_revenue * (1 + self.UG))
            self.max_total_revenue = max(self.max_total_revenue, self.total_revenue)
            self.PS = 0
            self.UG = 0
            return next_increase * 10
        else:
            return -99

    # 随机动作
    def random_action(self) -> int:
        if self.PS == 0:
            return np.random.choice([0, 1, 2], 1)[0]
        elif self.PS == 1:
            return np.random.choice([0, 0, 3], 1)[0]
        else:
            return np.random.choice([0, 0, 4], 1)[0]


if __name__ == "__main__":
    env = MyENV()
    state = env.reset()
    over = False
    while not over:
        action = env.random_action()
        env.check_print(action)
        state, reward, over, _ = env.step(action)
    print(f"total_revenue: {env.total_revenue}")
