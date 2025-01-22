import torch
from matplotlib import pyplot as plt
x = torch.tensor([5.0, 7.0, 12.0, 16.0, 20.0])
y = torch.tensor([40.0, 120.0, 180.0, 210.0, 240.0])

learning_rate = torch.tensor(0.001)
num_epochs = 100

class LinearRegression:
    def __init__(self):
        self.w = torch.rand([1], requires_grad=True)
        self.b = torch.rand([1], requires_grad=True)

    def forward(self, X):
        return self.w * X + self.b

    def update(self):
        self.w -= learning_rate * self.w.grad
        self.b -= learning_rate * self.b.grad

    def reset_grad(self):
        self.w.grad.zero_()
        self.b.grad.zero_()

def criterion(yj, yp):
    return (yp - yj)**2


model = LinearRegression()
loss_list = []


for epoch in range(num_epochs):
    loss = 0.0

    for j in range(len(x)):
        yp = model.forward(x[j])
        loss += criterion(yp, y[j])

    loss = loss / len(x)
    loss_list.append(loss.item())

    loss.backward()
    with torch.no_grad():
        model.update()

    model.reset_grad()

    print(f"Ep {epoch + 1}: w={model.w.item()} and b={model.b.item()}; loss = {loss.item()}")


plt.plot(loss_list)
plt.show()
