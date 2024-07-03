import os
import torch
import torch.nn as nn

import config


# 定义模型
class ActionModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.s = torch.nn.Sequential(
            torch.nn.Linear(config.prev_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
        )
        self.mu = torch.nn.Sequential(
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid(),
        )
        self.sigma = torch.nn.Sequential(
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, state):
        state = self.s(state)

        return self.mu(state), self.sigma(state).exp()


class SACModel:

    def __init__(self, mode="train"):
        self.modelName = config.modelName
        self.model_dir = config.model_dir
        self.create_model_dir()
        (
            self.model_action,
            self.model_value1,
            self.model_value2,
            self.model_value1_next,
            self.model_value2_next,
        ) = self.getModel(mode)

    def getModel(
        self, mode
    ) -> tuple[
        nn.Sequential, nn.Sequential, nn.Sequential, nn.Sequential, nn.Sequential
    ]:
        model_action = ActionModel()
        model_value1 = torch.nn.Sequential(
            torch.nn.Linear(config.prev_dim + 1, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2),
        )
        model_value2 = torch.nn.Sequential(
            torch.nn.Linear(config.prev_dim + 1, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2),
        )
        model_value1_next = torch.nn.Sequential(
            torch.nn.Linear(config.prev_dim + 1, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2),
        )
        model_value2_next = torch.nn.Sequential(
            torch.nn.Linear(config.prev_dim + 1, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2),
        )

        model_value1_next.load_state_dict(model_value1.state_dict())
        model_value2_next.load_state_dict(model_value2.state_dict())

        # 将模型移动到GPU（如果可用）
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", device)
        model_action.to(device)
        model_value1.to(device)
        model_value2.to(device)
        model_value1_next.to(device)
        model_value2_next.to(device)
        if mode == "eval":
            model_action.eval()  # 设置为评估模式
            model_value1.eval()  # 设置为评估模式
            model_value2.eval()  # 设置为评估模式
            model_value1_next.eval()  # 设置为评估模式
            model_value2_next.eval()  # 设置为评估模式
        else:
            model_action.train()  # 设置为训练模式
            model_value1.train()  # 设置为训练模式
            model_value2.train()  # 设置为训练模式
            model_value1_next.train()  # 设置为训练模式
            model_value2_next.train()  # 设置为训练模式
        return (
            model_action,
            model_value1,
            model_value2,
            model_value1_next,
            model_value2_next,
        )

    def checkModel(self, model: nn.Sequential) -> None:
        # 打印模型结构
        print(model)

    def create_model_dir(self) -> None:
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            print("{} 文件夹创建成功!".format(self.model_dir))

    def saveModel(
        self,
        model_action: nn.Sequential = None,
        model_value1: nn.Sequential = None,
        model_value2: nn.Sequential = None,
    ) -> None:
        if model_action is None:
            torch.save(
                self.model_action.state_dict(),
                f"{self.model_dir}/{self.modelName}_actor.pth",
            )
        else:
            torch.save(
                model_action.state_dict(),
                f"{self.model_dir}/{self.modelName}_actor.pth",
            )
        if model_value1 is None:
            torch.save(
                self.model_value1.state_dict(),
                f"{self.model_dir}/{self.modelName}_value1.pth",
            )
        else:
            torch.save(
                model_value1.state_dict(),
                f"{self.model_dir}/{self.modelName}_value1.pth",
            )
        if model_value2 is None:
            torch.save(
                self.model_value2.state_dict(),
                f"{self.model_dir}/{self.modelName}_value2.pth",
            )
        else:
            torch.save(
                model_value2.state_dict(),
                f"{self.model_dir}/{self.modelName}_value2.pth",
            )

    def loadModel(self) -> None:
        if os.path.exists(f"{self.model_dir}/{self.modelName}_actor.pth"):
            self.model_action.load_state_dict(
                torch.load(f"{self.model_dir}/{self.modelName}_actor.pth")
            )
            print(f"{self.modelName}_actor---模型加载成功!")
            self.checkModel(self.model_action)
        else:
            exit("model no exists.")
        if os.path.exists(f"{self.model_dir}/{self.modelName}_value1.pth"):
            self.model_value1.load_state_dict(
                torch.load(f"{self.model_dir}/{self.modelName}_value1.pth")
            )
            self.model_value1_next.load_state_dict(self.model_value1.state_dict())

            print(f"{self.modelName}_value1---模型加载成功!")
            self.checkModel(self.model_value1)
        else:
            exit("model no exists.")
        if os.path.exists(f"{self.model_dir}/{self.modelName}_value2.pth"):
            self.model_value2.load_state_dict(
                torch.load(f"{self.model_dir}/{self.modelName}_value2.pth")
            )
            self.model_value2_next.load_state_dict(self.model_value2.state_dict())
            print(f"{self.modelName}_value2---模型加载成功!")
            self.checkModel(self.model_value2)
        else:
            exit("model no exists.")


if __name__ == "__main__":
    model = SACModel()
    model.saveModel()
    model.loadModel()
