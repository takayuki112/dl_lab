import torch
import torch.nn as nn
from matplotlib import pyplot as plt

# Data
x = torch.tensor([1.0, 5.0, 10.0, 10.0, 25.0, 50.0, 70.0, 75.0, 100.0,]).view(-1, 1)
y = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]).view(-1, 1)

# Hyperparameters
learning_rate = 0.01
num_epochs = 500

# Model
class LogisticReg(nn.Module):
    def __init__(self):
        super(LogisticReg, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, X):
        return torch.sigmoid(self.linear(X))

model = LogisticReg()

criterion = nn.BCELoss()
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

    print(f"Ep {epoch + 1}: loss = {loss.item()}")

plt.plot(loss_list)
plt.show()

