import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 128 * 128, 2)  # 假设输入 256x256

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 16 * 128 * 128)
        x = self.fc1(x)
        return x

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")

    model = SimpleCNN().to(device)
    dummy_input = torch.randn(1, 1, 256, 256).to(device)  # 假数据
    output = model(dummy_input)

    print("Output:", output)
