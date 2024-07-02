import os
import torch
import torch.nn as nn


class DQNModel:

    def __init__(self, mode="train"):
        self.modelName = "Reinforce-(baselineRemoval)"
        self.model_dir = "save_model"
        self.create_model_dir()
        self.model, self.model_baseline = self.getModel(mode)

    def getModel(self, mode) -> tuple[nn.Sequential, nn.Sequential]:
        # 定义模型,评估状态下每个动作的价值
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2),
            torch.nn.Softmax(dim=1),
        )
        # 基线模型,评估state的价值
        model_baseline = torch.nn.Sequential(
            torch.nn.Linear(4, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )
        # 将模型移动到GPU（如果可用）
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", device)
        model.to(device)
        model_baseline.to(device)
        if mode == "eval":
            model.eval()  # 设置为评估模式
            model_baseline.eval()  # 设置为评估模式
        else:
            model.train()  # 设置为训练模式
            model_baseline.train()  # 设置为训练模式
        return model, model_baseline

    def checkModel(self) -> None:
        # 打印模型结构
        print(self.model)
        print(self.model_baseline)
        # 打印模型参数
        # for name, param in self.model.named_parameters():
        #     print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

    def create_model_dir(self) -> None:
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            print("{} 文件夹创建成功!".format(self.model_dir))

    def saveModel(
        self, model: nn.Sequential = None, model_baseline: nn.Sequential = None
    ) -> None:
        if model is None:
            torch.save(
                self.model.state_dict(), f"{self.model_dir}/{self.modelName}.pth"
            )
        else:
            torch.save(model.state_dict(), f"{self.model_dir}/{self.modelName}.pth")
        if model_baseline is None:
            torch.save(
                self.model_baseline.state_dict(),
                f"{self.model_dir}/{self.modelName}_baseline.pth",
            )
        else:
            torch.save(
                model_baseline.state_dict(),
                f"{self.model_dir}/{self.modelName}_baseline.pth",
            )

    def loadModel(self) -> None:
        if os.path.exists(f"{self.model_dir}/{self.modelName}.pth"):
            self.model.load_state_dict(
                torch.load(f"{self.model_dir}/{self.modelName}.pth")
            )
            self.model_baseline.load_state_dict(
                torch.load(f"{self.model_dir}/{self.modelName}_baseline.pth")
            )
            print(f"{self.modelName}---模型加载成功!")
            self.checkModel()
        else:
            exit("model no exists.")


if __name__ == "__main__":
    model = DQNModel()
    model.saveModel()
    model.loadModel()
