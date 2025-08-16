import torch

print('torch.cuda.is_available():', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device:', torch.cuda.get_device_name(0))
else:
    print('CUDA device not available')
