import torch

a = torch.tensor(2., requires_grad=True)
b = torch.tensor(3., requires_grad=True)

print("a, b = ", a, b)

x = 2*a + 3*b
y = 5*a*a + 3*b*b*b
z = 2*x + 3*y

print("x = ", x)
print("y = ", y)
print("z = ", z)

z.backward()
print("dz / da = ", a.grad)
print("dz / db = ", b.grad)

