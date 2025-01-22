import torch

scalar = torch.tensor([[7, 7],
                      [9, 10]])
print(scalar.shape)

random_t = torch.rand(size = (3, 4))
print("Random tensor = \n", random_t)

zero_t = torch.zeros(size = (3, 4))
print("Zero tensor = \n", zero_t)
print("Datatype = ", zero_t.dtype)

zero_to_ten = torch.arange(start=0, end=10, step=1)
print("0 -> 10 =\n", zero_to_ten)

print("\nAddition 10 = \n", scalar+10)

