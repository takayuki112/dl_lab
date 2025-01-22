import torch

# Q7 - create two and move to gpu
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

t1 = torch.randn((2, 3))
print(t1, t1.device)

t1 = t1.to("cuda:0")
# t1 = t1.cuda()
print(t1, t1.device)

t2 = torch.randn((2, 3))
t2 = t2.cuda()
print(t2, t2.device)

# Q8 - matmul
res = torch.matmul(t1, t2.T)
print(res, res.device)

# Q9 - Min and max
print("Max = ", res.max())
print("Min = ", res.min())

# Q10 - Min and Max Index
print("Max Index", res.argmax())
print("Min Index", res.argmin())


# Q11 - something
torch.manual_seed(7)
ti = torch.randn((1, 1, 1, 10))
print(ti, "\nBefore - ", ti.shape)
ti = ti.squeeze(0, 1, 2)
print(ti, "\nAfter - ", ti.shape)
