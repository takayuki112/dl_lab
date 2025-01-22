import torch
from matplotlib import pyplot as plt

x = torch.tensor([2.0, 4.0])
y = torch.tensor([20.0, 40.0])

# Parameters w, b (are single numbers); For Model yp = wx + b
w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

print(f"The initial values of the parameters w, b are = {w} and {b}")

lr = torch.tensor(0.01)
num_epochs = 200

loss_values = []

for epoch in range(num_epochs):
    loss = 0.0

    for j in range(len(x)):
        yp = w * x[j] + b
        loss += (yp - y[j]) ** 2

    loss = loss / len(x)
    loss_values.append(loss.item())

    loss.backward()
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

    w.grad.zero_()
    b.grad.zero_()

    print(f"Ep {epoch + 1}: w={w.item()} and b={b.item()}; loss = {loss.item()}")

plt.plot(loss_values)
plt.show()



