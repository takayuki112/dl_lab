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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FeedForward().to(device)
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

#op -
# /home/student/myenv/bin/python /home/student/Documents/220962366/week4/q4_mnist.py
# /home/student/myenv/lib/python3.12/site-packages/torch/cuda/__init__.py:129: UserWarning: CUDA initialization: The NVIDIA driver on your system is too old (found version 11040). Please update your GPU driver by downloading and installing a new version from the URL: http://www.nvidia.com/Download/index.aspx Alternatively, go to: https://pytorch.org to install a PyTorch version that has been compiled with your version of the CUDA driver. (Triggered internally at ../c10/cuda/CUDAFunctions.cpp:108.)
#   return torch._C._cuda_getDeviceCount() > 0
# Type is =  <class 'torchvision.datasets.mnist.MNIST'>
# Train Dataset size =  60000
# Initializing training on dev =  cpu
# Epoch 1/15, Loss: 1.6905, Accuracy: 78.36%
# Epoch 2/15, Loss: 1.6052, Accuracy: 85.67%
# Epoch 3/15, Loss: 1.5476, Accuracy: 91.58%
# Epoch 4/15, Loss: 1.5086, Accuracy: 95.40%
# Epoch 5/15, Loss: 1.4996, Accuracy: 96.28%
# Epoch 6/15, Loss: 1.4946, Accuracy: 96.75%
# Epoch 7/15, Loss: 1.4917, Accuracy: 97.02%
# Epoch 8/15, Loss: 1.4897, Accuracy: 97.22%
# Epoch 9/15, Loss: 1.4868, Accuracy: 97.48%
# Epoch 10/15, Loss: 1.4845, Accuracy: 97.72%
# Epoch 11/15, Loss: 1.4825, Accuracy: 97.91%
# Epoch 12/15, Loss: 1.4828, Accuracy: 97.86%
# Epoch 13/15, Loss: 1.4807, Accuracy: 98.11%
# Epoch 14/15, Loss: 1.4799, Accuracy: 98.16%
# Epoch 15/15, Loss: 1.4788, Accuracy: 98.28%