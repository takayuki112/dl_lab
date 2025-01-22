import torch
import torch.nn as nn
from matplotlib import pyplot as plt

# Data
x1 = torch.tensor([3.0, 4.0, 5.0, 6.0, 2.0]).view(-1, 1)
x2 = torch.tensor([8.0, 5.0, 7.0, 3.0, 1.0]).view(-1, 1)
x = torch.hstack([x1, x2])
y = torch.tensor([-3.7, 3.5, 2.5, 11.5, 5.7]).view(-1, 1)
# Hyperparameters
learning_rate = 0.01
num_epochs = 800

# Model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(2, 1)

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

    print(f"Ep {epoch + 1}: loss = {loss.item()}")

xn = torch.tensor([3.0, 2.0])
print("Prediction for the values x1 = 3 and x2 = 2 is: ", model(xn))

plt.plot(loss_list)
plt.show()
