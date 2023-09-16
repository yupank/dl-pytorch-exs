import torch
import numpy as np

 # just testing the torch and tensors

x = torch.tensor([[1,2,3],[4,5,6]])
# print((x[1][0:2]).numpy())
# print(x.view(-1))
# print(x.view(3,2))
# print(x.view(6,1))

torch.manual_seed(13)
y = torch.rand([2,3])

f = 2*x + y
print(f)
# print(f.transpose_(1,0))
# y.add_(x*10)
print(y)

a = torch.rand(1,2,3,4)
# print(a)
# print(a.transpose(0,3).transpose(1,2))
# print(a.permute(3,2,1,0))