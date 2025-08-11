import torch
import torch.nn as nn

# 1. 定义一个非常简单的 CNN 模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 128 * 128, 2)  # 输入是 256x256 的单通道图像

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 16 * 128 * 128)
        x = self.fc1(x)
        return x

if __name__ == "__main__":
    # 2. 检测是否有 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前运行设备: {device}")

    # 3. 创建模型并放到 GPU
    model = SimpleCNN().to(device)

    # 4. 创建假数据（1 张 256x256 的 CT 灰度图）
    dummy_input = torch.randn(1, 1, 256, 256).to(device)

    # 5. 前向推理
    output = model(dummy_input)

    # 6. 输出结果
    print("模型输出:", output)
