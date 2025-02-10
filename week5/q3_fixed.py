import torch
import torch.nn as nn

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

from matplotlib import pyplot as plt

transform = transforms.Compose([transforms.ToTensor()])

mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)  # Added download=True
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)  # Added download=True

print("Shape = ", mnist_train[0][0].shape)
print("Type = ", type(mnist_train[0]))
print("Training size = ", len(mnist_train[0]))
print("Testing size = ", len(mnist_test))

batch_size = 64 # Increased batch size

train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size)


class CNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),  # Input channels = 1, Output channels = 32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Reduced pooling
            nn.Conv2d(32, 64, kernel_size=3), # Input channels = 32, Output channels = 64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Reduced pooling
            nn.Flatten(start_dim=1) # Flatten the output for the linear layer
        )

        # Calculate the input size for the first linear layer based on output of convolutional layers
        self.classification_head = nn.Sequential(
            nn.Linear(64 * 5 * 5, 128, bias=True),  # Adjust input size after flattening
            nn.ReLU(),
            nn.Linear(128, 10, bias=True) # Output dimension 10 for digits 0-9
        )

    def forward(self, x):
        features = self.net(x)
        return self.classification_head(features)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device = ", device)

model = CNNClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=0.001) # Reduce the learning rate

print("Number of parameters = ", sum(p.numel() for p in model.parameters()))


def train_one_epoch(epoch_idx):
    model.train() # Set to train mode
    total_loss = 0
    total_correct = 0
    total_samples = 0
    for i, data in enumerate(train_loader):
        inputs, labels = data[0].to(device), data[1].to(device) # Move data to device

        optim.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optim.step()

        total_loss += loss.item() * inputs.size(0) # weighted loss by the number of instances
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    accuracy = (total_correct / total_samples) * 100
    return total_loss / total_samples, accuracy


epochs = 10
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