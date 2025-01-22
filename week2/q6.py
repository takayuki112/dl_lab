import torch

x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)
z = torch.tensor(5.0, requires_grad=True)
# forward pass
a = 2*x
b = torch.sin(y)
c = a/b

d = c*z
e = torch.log(d+1)
f = torch.tanh(e)

print("Intermediate values after forward pass")
print("a = ", a)
print("b = ", b)
print("c = ", c)
print("d = ", d)
print("e = ", e)
print("f = ", f)

# Manual backward pass
print("\nManual backward pass")
df_de = 1 - torch.tanh(e)**2
de_dd = 1/(d+1)
dd_dz = c
dd_dc = z
dc_db = a/b**2
dc_da = 1/b
db_dy = torch.cos(y)
da_dx = 2

print("df_de = ", df_de)
print("de_dd = ", de_dd)
print("dd_dz = ", dd_dz)
print("dd_dc = ", dd_dc)
print("dc_db = ", dc_db)
print("dc_da = ", dc_da)
print("db_dy = ", db_dy)
print("da_dx = ", da_dx)

df_dx = df_de * de_dd * dd_dc * dc_da * da_dx
print("Manually, df_dx = ", df_dx)

f.backward()
print("By Autograd, df_dx = ", x.grad)




