import torch
from matplotlib import pyplot as plt

x = torch.tensor( [12.4, 14.3, 14.5, 14.9, 16.1, 16.9, 16.5, 15.4, 17.0, 17.9, 18.8, 20.3, 22.4, 19.4, 15.5, 16.7, 17.3, 18.4, 19.2, 17.4, 19.5, 19.7, 21.2])
y = torch.tensor( [11.2, 12.5, 12.7, 13.1, 14.1, 14.8, 14.4, 13.4, 14.9, 15.6, 16.4, 17.7, 19.6, 16.9, 14.0, 14.6, 15.1, 16.1, 16.8, 15.2, 17.0, 17.2, 18.6])

# Parameters w, b (are single numbers); For Model yp = wx + b
w = torch.rand([1], requires_grad = True)
b = torch.rand([1], requires_grad = True)

print(f"The initial values of the parameters w, b are = {w} and {b}")

lr = torch.tensor(0.001)
num_epochs = 100

loss_values = []

for epoch in range(num_epochs):
    loss = 0.0

    for j in range(len(x)):
        yp = w * x[j] + b
        loss += (yp - y[j])**2

    loss = loss/len(x)
    loss_values.append(loss.item())

    loss.backward()
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

    w.grad.zero_()
    b.grad.zero_()

    print(f"Ep {epoch + 1}: w={w.item()} and b={b.item()}; loss = {loss.item()}")

plt.plot(loss_values)
plt.title("Loss vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid()
plt.legend(["Loss"])
plt.show()



                  