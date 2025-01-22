import torch
import numpy as np

# Question 2 - permute
t1 = torch.randn(2, 3, 4)
print(t1.size())
t2 = t1.permute(2, 1, 0)
print("After permute = \n", t2.size())

# Question 3 - Indexing
print(t1)
print(t1[0, 1, 1])
print(t1[0, 1, 2])

#Question 4 - Numpy and tensors
n1 = np.array([1, 2, 3])
print("N1 is currently of type = ", type(n1))
n1 = torch.from_numpy(n1)
print("N1 is currently of type = ", type(n1))
n1 = n1.cpu().numpy()
print("N1 is currently of type = ", type(n1))

#Question 5 - random 7x7
r1 = torch.randn((7, 7))
print(r1.shape)
print(r1)

# Question 6 - matrix-multi
r2 = torch.randn((1, 7))
res = torch.matmul(r1, r2.T)
print(res.shape)
print(res)
