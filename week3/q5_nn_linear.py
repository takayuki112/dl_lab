import torch
import torch.nn as nn
from matplotlib import pyplot as plt

# Data
x = torch.tensor([12.4, 14.3, 14.5, 14.9, 16.1, 16.9, 16.5, 15.4, 17.0, 17.9, 18.8, 20.3, 22.4, 19.4, 15.5, 16.7, 17.3, 18.4, 19.2, 17.4, 19.5, 19.7, 21.2]).view(-1, 1)
y = torch.tensor([11.2, 12.5, 12.7, 13.1, 14.1, 14.8, 14.4, 13.4, 14.9, 15.6, 16.4, 17.7, 19.6, 16.9, 14.0, 14.6, 15.1, 16.1, 16.8, 15.2, 17.0, 17.2, 18.6]).view(-1, 1)

# Hyperparameters
learning_rate = 0.001
num_epochs = 100

# Model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, X):
        return self.linear(X)

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

    print(f"Ep {epoch + 1}: w={model.linear.weight.item()} and b={model.linear.bias.item()}; loss = {loss.item()}")

plt.plot(loss_list)
plt.show()
