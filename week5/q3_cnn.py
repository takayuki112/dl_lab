import torch
import torch.nn as nn

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

from matplotlib import pyplot as plt

transform = transforms.Compose([transforms.ToTensor()])

mnist_train = datasets.MNIST(root='./data', train=True, transform=transform)
mnist_test = datasets.MNIST(root='./data', train=False, transform=transform)

print("Shape = ", mnist_train[0][0].shape)
print("Type = ", type(mnist_train[0]))
print("Training size = ", len(mnist_train[0]))
print("Testing size = ", len(mnist_test))

batch_size = 1

train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size)


class CNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential( nn.Conv2d(1, 64, kernel_size=3),
                                  nn.ReLU(),
                                  nn.MaxPool2d((2, 2), stride=2),
                                  nn.Conv2d(64, 128, kernel_size=3),
                                  nn.ReLU(),
                                  nn.MaxPool2d((2, 2), stride=2),
                                  nn.Conv2d(128, 64, kernel_size=3),
                                  nn.ReLU(),
                                  nn.MaxPool2d((2, 2), stride=2)
                                )
        self.classification_head = nn.Sequential( nn.Linear(64, 20, bias=True),
                                                  nn.ReLU(),
                                                  nn.Linear(20, 10, bias=True)
                                                )
    def forward(self, x):
        features = self.net(x)
        return self.classification_head(features)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=0.01)

print("Number of parameters = ", sum(p.numel() for p in model.parameters()))


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


epochs = 15
loss_values = []
print("Initializing training on dev = ", device)
for epoch in range(epochs):
    avg_loss, accuracy = train_one_epoch(epoch)
    loss_values.append(avg_loss)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

plt.plot(loss_values)
plt.grid()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()