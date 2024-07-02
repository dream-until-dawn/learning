import os
import torch
import torch.nn as nn


# 定义模型
class ActionModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.s = torch.nn.Sequential(
            torch.nn.Linear(3, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
            torch.nn.Tanh(),
        )

    def forward(self, state):
        return self.s(state)


class DQNModel:

    def __init__(self, mode="train"):
        self.modelName = "TD3"
        self.model_dir = "save_model"
        self.create_model_dir()
        (
            self.model_action,
            self.model_action_delay,
            self.model_value1,
            self.model_value1_delay,
            self.model_value2,
            self.model_value2_delay,
        ) = self.getModel(mode)

    def getModel(self, mode) -> tuple[
        nn.Sequential,
        nn.Sequential,
        nn.Sequential,
        nn.Sequential,
        nn.Sequential,
        nn.Sequential,
    ]:
        model_action = ActionModel()
        model_action_delay = ActionModel()
        model_value1 = torch.nn.Sequential(
            torch.nn.Linear(4, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )
        model_value2 = torch.nn.Sequential(
            torch.nn.Linear(4, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )
        model_value1_delay = torch.nn.Sequential(
            torch.nn.Linear(4, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )
        model_value2_delay = torch.nn.Sequential(
            torch.nn.Linear(4, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )

        model_action_delay.load_state_dict(model_action.state_dict())
        model_value1_delay.load_state_dict(model_value1.state_dict())
        model_value2_delay.load_state_dict(model_value2.state_dict())

        # 将模型移动到GPU（如果可用）
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", device)
        model_action.to(device)
        model_action_delay.to(device)
        model_value1.to(device)
        model_value1_delay.to(device)
        model_value2.to(device)
        model_value2_delay.to(device)
        if mode == "eval":
            model_action.eval()  # 设置为评估模式
            model_action_delay.eval()  # 设置为评估模式
            model_value1.eval()  # 设置为评估模式
            model_value1_delay.eval()  # 设置为评估模式
            model_value2.eval()  # 设置为评估模式
            model_value2_delay.eval()  # 设置为评估模式
        else:
            model_action.train()  # 设置为训练模式
            model_action_delay.train()  # 设置为训练模式
            model_value1.train()  # 设置为训练模式
            model_value1_delay.train()  # 设置为训练模式
            model_value2.train()  # 设置为训练模式
            model_value2_delay.train()  # 设置为训练模式
        return (
            model_action,
            model_action_delay,
            model_value1,
            model_value1_delay,
            model_value2,
            model_value2_delay,
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
            self.model_value1_delay.load_state_dict(self.model_value1.state_dict())

            print(f"{self.modelName}_value1---模型加载成功!")
            self.checkModel(self.model_value1)
        else:
            exit("model no exists.")
        if os.path.exists(f"{self.model_dir}/{self.modelName}_value2.pth"):
            self.model_value2.load_state_dict(
                torch.load(f"{self.model_dir}/{self.modelName}_value2.pth")
            )
            self.model_value2_delay.load_state_dict(self.model_value2.state_dict())
            print(f"{self.modelName}_value2---模型加载成功!")
            self.checkModel(self.model_value2)
        else:
            exit("model no exists.")


if __name__ == "__main__":
    model = DQNModel()
    model.saveModel()
    model.loadModel()
