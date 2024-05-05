import torch

if torch.cuda.is_available():
    device_index = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device_index)
    bf16_support = torch.cuda.is_bf16_supported()
    device_capability=torch.cuda.get_device_capability(device_index)
    print(f"设备名称: {device_name}")
    print(f"设备能力: {device_capability}")
    print(f"设备支持bfloat16: {'是' if bf16_support else '否'}")
else:
    print("CUDA不可用")