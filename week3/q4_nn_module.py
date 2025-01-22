import torch
import torch.nn as nn
from matplotlib import pyplot as plt

# Data
x = torch.tensor([5.0, 7.0, 12.0, 16.0, 20.0]).view(-1, 1)  # Reshaping for a single feature
y = torch.tensor([40.0, 120.0, 180.0, 210.0, 240.0]).view(-1, 1)

# Hyperparameters
learning_rate = 0.001
num_epochs = 100

# Model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.w = nn.Parameter(torch.rand([1]))
        self.b = nn.Parameter(torch.rand([1]))

    def forward(self, x):
        return self.w * x + self.b

    # everything else like gradients and all already taken care of by nn module

model = LinearRegressionModel()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training loop
loss_list = []
for epoch in range(num_epochs):
    model.train()

    # fwd pass
    yp = model(x)

    # loss
    loss = criterion(yp, y)
    loss_list.append(loss.item())

    # backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Ep {epoch + 1}: w={model.w.item()} and b={model.b.item()}; loss = {loss.item()}")

plt.plot(loss_list)
plt.show()
