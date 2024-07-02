import os
import torch
import torch.nn as nn


class DQNModel:

    def __init__(self, mode="train"):
        self.modelName = "Reinforce"
        self.model_dir = "save_model"
        self.create_model_dir()
        self.model = self.getModel(mode)

    def getModel(self, mode) -> nn.Sequential:
        # 定义模型,评估状态下每个动作的价值
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2),
            torch.nn.Softmax(dim=1),
        )
        # 将模型移动到GPU（如果可用）
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", device)
        model.to(device)
        if mode == "eval":
            model.eval()  # 设置为评估模式
        else:
            model.train()  # 设置为训练模式
        return model

    def checkModel(self) -> None:
        # 打印模型结构
        print(self.model)
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
                torch.load(f"{self.model_dir}/{self.modelName}.pth")
            )
            print(f"{self.modelName}---模型加载成功!")
            self.checkModel()
        else:
            exit("model no exists.")


if __name__ == "__main__":
    model = DQNModel()
    model.loadModel()
