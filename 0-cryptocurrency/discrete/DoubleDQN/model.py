import os
import torch
import torch.nn as nn

import config


class DQNModel:

    def __init__(self, mode="train"):
        self.modelName = config.modelName
        self.model_dir = config.model_dir
        self.create_model_dir()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.model_delay = self.getModel(mode)

    def getModel(self, mode) -> tuple[nn.Sequential, nn.Sequential]:
        # 定义模型,评估状态下每个动作的价值
        model = torch.nn.Sequential(
            torch.nn.Linear(config.prev_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 5),
        )
        # 延迟更新的模型,用于计算target
        model_delay = torch.nn.Sequential(
            torch.nn.Linear(config.prev_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 5),
        )
        model_delay.load_state_dict(model.state_dict())
        # 将模型移动到GPU（如果可用）
        print("Using device:", self.device)
        model.to(self.device)
        model_delay.to(self.device)
        if mode == "eval":
            model.eval()  # 设置为评估模式
            model_delay.eval()  # 设置为评估模式
        else:
            model.train()  # 设置为训练模式
            model_delay.train()  # 设置为训练模式
        return model, model_delay

    def checkModel(self) -> None:
        # 打印模型结构
        print(self.model)
        print(self.model_delay)
        # 打印模型参数
        # for name, param in self.model.named_parameters():
        #     print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

    def create_model_dir(self) -> None:
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            print("{} 文件夹创建成功!".format(self.model_dir))

    def saveModel(self, model: nn.Sequential = None) -> None:
        if model is None:
            torch.save(
                self.model.state_dict(), f"{self.model_dir}/{self.modelName}.pth"
            )
        else:
            torch.save(model.state_dict(), f"{self.model_dir}/{self.modelName}.pth")

    def loadModel(self) -> None:
        if os.path.exists(f"{self.model_dir}/{self.modelName}.pth"):
            self.model.load_state_dict(
                torch.load(
                    f"{self.model_dir}/{self.modelName}.pth",
                    map_location=torch.device("cpu"),
                )
            )
            self.model_delay.load_state_dict(self.model.state_dict())
            self.model.to(self.device)
            self.model_delay.to(self.device)
            print(f"{self.modelName}---模型加载成功!")
            self.checkModel()
        else:
            exit("model no exists.")


if __name__ == "__main__":
    model = DQNModel()
    model.saveModel()
    model.loadModel()
