import os
import torch
import torch.nn as nn


class DQNModel:

    def __init__(self, mode="train"):
        self.modelName = "A2C"
        self.model_dir = "save_model"
        self.create_model_dir()
        self.model_actor, self.model_critic, self.model_critic_delay = self.getModel(
            mode
        )

    def getModel(self, mode) -> tuple[nn.Sequential, nn.Sequential, nn.Sequential]:
        # 演员模型,计算每个动作的概率
        model_actor = torch.nn.Sequential(
            torch.nn.Linear(4, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2),
            torch.nn.Softmax(dim=1),
        )

        # 评委模型,计算每个状态的价值
        model_critic = torch.nn.Sequential(
            torch.nn.Linear(4, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )

        model_critic_delay = torch.nn.Sequential(
            torch.nn.Linear(4, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )

        model_critic_delay.load_state_dict(model_critic.state_dict())

        # 将模型移动到GPU（如果可用）
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", device)
        model_actor.to(device)
        model_critic.to(device)
        model_critic_delay.to(device)
        if mode == "eval":
            model_actor.eval()  # 设置为评估模式
            model_critic.eval()  # 设置为评估模式
            model_critic_delay.eval()  # 设置为评估模式
        else:
            model_actor.train()  # 设置为训练模式
            model_critic.train()  # 设置为训练模式
            model_critic_delay.train()  # 设置为训练模式
        return model_actor, model_critic, model_critic_delay

    def checkModel(self) -> None:
        # 打印模型结构
        print(self.model_actor)
        print(self.model_critic)
        print(self.model_critic_delay)

    def create_model_dir(self) -> None:
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            print("{} 文件夹创建成功!".format(self.model_dir))

    def saveModel(
        self, model_actor: nn.Sequential = None, model_critic: nn.Sequential = None
    ) -> None:
        if model_actor is None:
            torch.save(
                self.model_actor.state_dict(),
                f"{self.model_dir}/{self.modelName}_actor.pth",
            )
        else:
            torch.save(
                model_actor.state_dict(), f"{self.model_dir}/{self.modelName}_actor.pth"
            )
        if model_critic is None:
            torch.save(
                self.model_critic.state_dict(),
                f"{self.model_dir}/{self.modelName}_critic.pth",
            )
        else:
            torch.save(
                model_critic.state_dict(),
                f"{self.model_dir}/{self.modelName}_critic.pth",
            )

    def loadModel(self) -> None:
        if os.path.exists(f"{self.model_dir}/{self.modelName}_actor.pth"):
            self.model_actor.load_state_dict(
                torch.load(f"{self.model_dir}/{self.modelName}_actor.pth")
            )
            print(f"{self.modelName}_actor---模型加载成功!")
            self.checkModel()
        else:
            exit("model no exists.")
        if os.path.exists(f"{self.model_dir}/{self.modelName}_critic.pth"):
            self.model_critic.load_state_dict(
                torch.load(f"{self.model_dir}/{self.modelName}_critic.pth")
            )
            self.model_critic_delay.load_state_dict(self.model_critic.state_dict())
            print(f"{self.modelName}_critic---模型加载成功!")
            self.checkModel()
        else:
            exit("model no exists.")


if __name__ == "__main__":
    model = DQNModel()
    model.saveModel()
    model.loadModel()
