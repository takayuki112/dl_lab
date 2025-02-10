import torch
from matplotlib import pyplot as plt

x = torch.tensor([2.0, 4.0])
y = torch.tensor([20.0, 40.0])

# Parameters w, b (are single numbers); For Model yp = wx + b
w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

print(f"The initial values of the parameters w, b are = {w} and {b}")

lr = torch.tensor(0.001)
num_epochs = 2

loss_values = []

for epoch in range(num_epochs):
    loss = 0.0
    
    dw = 0.0
    db = 0.0
    
    for j in range(len(x)):
        yp = w * x[j] + b
        loss += (yp - y[j]) ** 2
        
        dw += 2 * (yp - y[j]) * x[j]
        db += 2 * (yp - y[j])

    loss_values.append(loss.item())

    loss.backward()
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
    
    print(f"\nEp {epoch + 1}: w={w.item()} and b={b.item()}; loss = {loss.item()}")
    print(f"Analytically:\t dw = {dw.item()} and db = {db.item()}")
    print("Autograd:\t w.grad = ", w.grad.item(), "b.grad = ", b.grad.item())
    
    
    w.grad.zero_()
    b.grad.zero_()

plt.plot(range(1, 3), loss_values)
plt.title("Loss vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid()
plt.legend(["Loss"])
plt.show()



