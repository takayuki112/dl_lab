import torch

x = torch.tensor(5., requires_grad=True)

f = torch.exp(-(torch.square(x)) - 2*x - torch.sin(x))

f.backward()

print("df / dx = ", x.grad)