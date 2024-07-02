import os
import torch
import torch.nn as nn


class DQNModel(torch.nn.Module):
    def __init__(self, mode="train"):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(4, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
        )
        self.fc_action = torch.nn.Linear(64, 2)
        self.fc_state = torch.nn.Linear(64, 1)

        self.modelName = "DuelingDQN"
        self.model_dir = "save_model"
        self.create_model_dir()
        self.setModel(mode)

    def setModel(self, mode) -> nn.Sequential:
        # 将模型移动到GPU（如果可用）
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", device)
        self.fc.to(device)
        if mode == "eval":
            self.fc.eval()  # 设置为评估模式
        else:
            self.fc.train()  # 设置为训练模式
        return self.fc

    def checkModel(self) -> None:
        # 打印模型结构
        print(self.fc)

    def create_model_dir(self) -> None:
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            print("{} 文件夹创建成功!".format(self.model_dir))

    def saveModel(self, model: nn.Sequential = None) -> None:
        if model is None:
            torch.save(self.state_dict(), f"{self.model_dir}/{self.modelName}.pth")
        else:
            torch.save(model.state_dict(), f"{self.model_dir}/{self.modelName}.pth")

    def loadModel(self) -> None:
        if os.path.exists(f"{self.model_dir}/{self.modelName}.pth"):
            self.load_state_dict(
                torch.load(f"{self.model_dir}/{self.modelName}.pth")
            )
            print(f"{self.modelName}---模型加载成功!")
            self.checkModel()
        else:
            exit("model no exists.")

    def forward(self, state):
        state = self.fc(state)

        # 评估state的价值
        value_state = self.fc_state(state)

        # 每个state下每个action的价值
        value_action = self.fc_action(state)

        # 综合以上两者计算最终的价值,action去均值是为了数值稳定
        return value_state + value_action - value_action.mean(dim=-1, keepdim=True)


if __name__ == "__main__":
    model = DQNModel()
    model.loadModel()
    model_delay = DQNModel()
    model_delay.load_state_dict(model.state_dict())
