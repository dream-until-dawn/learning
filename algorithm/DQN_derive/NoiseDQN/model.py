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
        # 输出层参数的均值和标准差
        self.weight_mean = torch.nn.Parameter(torch.randn(64, 2))
        self.weight_std = torch.nn.Parameter(torch.randn(64, 2))

        self.bias_mean = torch.nn.Parameter(torch.randn(2))
        self.bias_std = torch.nn.Parameter(torch.randn(2))

        self.modelName = "NoiseDQN"
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
            self.load_state_dict(torch.load(f"{self.model_dir}/{self.modelName}.pth"))
            print(f"{self.modelName}---模型加载成功!")
            self.checkModel()
        else:
            exit("model no exists.")

    def forward(self, state):
        state = self.fc(state)

        # 正态分布投影,获取输出层的参数
        weight = self.weight_mean + torch.randn(64, 2) * self.weight_std
        bias = self.bias_mean + torch.randn(2) * self.bias_std

        # 运行模式下不需要随机性
        if not self.training:
            weight = self.weight_mean
            bias = self.bias_mean

        # 计算输出
        return state.matmul(weight) + bias


if __name__ == "__main__":
    model = DQNModel()
    model_delay = DQNModel()
    model_delay.load_state_dict(model.state_dict())
