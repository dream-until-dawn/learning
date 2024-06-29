import os
import torch
import torch.nn as nn


class DQNModel:

    def __init__(self, mode="train"):
        self.modelName = "PPO-discrete"
        self.model_dir = "save_model"
        self.create_model_dir()
        self.model_action, self.model_value = self.getModel(mode)

    def getModel(self, mode) -> tuple[nn.Sequential, nn.Sequential]:
        # 演员模型,计算每个动作的概率
        model_action = torch.nn.Sequential(
            torch.nn.Linear(4, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2),
            torch.nn.Softmax(dim=1),
        )

        model_value = torch.nn.Sequential(
            torch.nn.Linear(4, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )

        # 将模型移动到GPU（如果可用）
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", device)
        model_action.to(device)
        model_value.to(device)
        if mode == "eval":
            model_action.eval()  # 设置为评估模式
            model_value.eval()  # 设置为评估模式
        else:
            model_action.train()  # 设置为训练模式
            model_value.train()  # 设置为训练模式
        return model_action, model_value

    def checkModel(self, model: nn.Sequential) -> None:
        # 打印模型结构
        print(model)

    def create_model_dir(self) -> None:
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            print("{} 文件夹创建成功!".format(self.model_dir))

    def saveModel(
        self, model_action: nn.Sequential = None, model_value: nn.Sequential = None
    ) -> None:
        if model_action is None:
            torch.save(
                self.model_action.state_dict(),
                f"{self.model_dir}/{self.modelName}_actor.pth",
            )
        else:
            torch.save(
                model_action.state_dict(), f"{self.model_dir}/{self.modelName}_actor.pth"
            )
        if model_value is None:
            torch.save(
                self.model_value.state_dict(),
                f"{self.model_dir}/{self.modelName}_value.pth",
            )
        else:
            torch.save(
                model_value.state_dict(),
                f"{self.model_dir}/{self.modelName}_value.pth",
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
        if os.path.exists(f"{self.model_dir}/{self.modelName}_value.pth"):
            self.model_value.load_state_dict(
                torch.load(f"{self.model_dir}/{self.modelName}_value.pth")
            )
            print(f"{self.modelName}_value---模型加载成功!")
            self.checkModel(self.model_value)
        else:
            exit("model no exists.")


if __name__ == "__main__":
    model = DQNModel()
    model.saveModel()
    model.loadModel()
