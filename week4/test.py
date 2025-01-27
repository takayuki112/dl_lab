import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import Sequential

import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# Apply the ToTensor transformation
transform = transforms.Compose([transforms.ToTensor()])

mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

print("Type is = ", type(mnist_train))
print("Train Dataset size = ", len(mnist_train))

batch_size = 1

train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size)


class FeedForward(nn.Module):
    def __init__(self):
        super(FeedForward, self).__init__()
        self.net = Sequential(
            nn.Linear(28 * 28, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the image
        return self.net(x)


model = FeedForward()
criterion = nn.CrossEntropyLoss()
optim = torch.optim.SGD(model.parameters(), lr=0.01)


def train_one_epoch(epoch_idx):
    total_loss = 0
    total_correct = 0
    total_samples = 0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        optim.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optim.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    accuracy = (total_correct / total_samples) * 100
    return total_loss / len(train_loader), accuracy


epochs = 5
loss_values = []
print("Initializing training")
for epoch in range(epochs):
    avg_loss, accuracy = train_one_epoch(epoch)
    loss_values.append(avg_loss)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

plt.plot(loss_values)
plt.grid()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()