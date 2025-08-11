import torch
print(torch.__version__)          # 查看 PyTorch 版本
print(torch.cuda.is_available())  # 检查 CUDA 是否可用
print(torch.cuda.current_device())  # 获取当前设备编号
print(torch.cuda.get_device_name(0))  # 获取显卡名称
