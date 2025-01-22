import torch


# Question 1
t1 = torch.arange(0, 10, 1)
print(t1.shape)

reshaped = t1.reshape((5, 2))
print("Reshaped = \n", reshaped.shape)

t2 = torch.zeros((5, 2))
print(t2)

stacked = torch.hstack((reshaped, t2))
print("Stacked = \n", stacked)

unsqueezed = stacked.unsqueeze(1)
print("Original Shape was = ", stacked.shape)
print("After Unsqueezing = ", unsqueezed.shape)
print(unsqueezed)

squeezed = unsqueezed.squeeze(1)
print("Again got back to shape = ", squeezed.shape)


