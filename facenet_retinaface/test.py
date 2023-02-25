import torch
# 若正常则静默
a = torch.tensor(1.)
# 若正常则静默
print(a.cuda())
# 若正常则返回 tensor(1., device='cuda:0')
from torch.backends import cudnn
# 若正常则静默
print(cudnn.is_available())
# 若正常则返回 True
print(cudnn.is_acceptable(a.cuda()))
print(torch.cuda.is_available())
print(torch.cuda.device_count())