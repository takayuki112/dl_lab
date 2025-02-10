import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Hyperparameters
batch_size = 64
learning_rate = 0.001
epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data loading and preprocessing
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


# Base CNN Model
class CNN(nn.Module):
    def __init__(self, filters1=32, filters2=64, linear1=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, filters1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(filters1, filters2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(start_dim=1)
        )
        self.classification_head = nn.Sequential(
            nn.Linear((filters2 * 7 * 7), linear1),
            nn.ReLU(),
            nn.Linear(linear1, 10)
        )

    def forward(self, x):
        return self.classification_head(self.net(x))


# Training and Evaluation Function (Combined)
def train_evaluate(model, train_loader, test_loader, learning_rate, epochs, device):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            correct = sum((model(images.to(device)).argmax(1) == labels.to(device)).sum().item() for images, labels in test_loader)
            accuracy = 100 * correct / len(test_loader.dataset)
            print(f'Epoch [{epoch+1}/{epochs}], Accuracy: {accuracy:.2f}%')
        model.train()
    return accuracy


# Function to count parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


# Model configurations
model_configs = {   
    "Base": (32, 64, 128),
    "Mod1": (16, 32, 64),   # Reduced filters, smaller linear layer
    "Mod2": (32, 64, 64),   # Smaller linear layer
    "Mod3": (16, 32, 32),   # Reduced filters, smaller linear layer, fewer layers (implicitly)
}

# Train, evaluate, and collect results
results = {}
base_params = None  # Store base model parameters
for name, (filters1, filters2, linear1) in model_configs.items():
    model = CNN(filters1, filters2, linear1).to(device)
    params = count_parameters(model)
    accuracy = train_evaluate(model, train_loader, test_loader, learning_rate, epochs, device)
    results[name] = {"params": params, "accuracy": accuracy}
    if name == "Base":
        base_params = params


# Calculate parameter drops
param_drops = {name: (1 - (data["params"] / base_params)) * 100 if base_params else 0 for name, data in results.items()}


# Print results
print("Results:")
for name, data in results.items():
    print(f"{name}: Parameters = {data['params']}, Param Drop = {param_drops[name]:.2f}%, Accuracy = {data['accuracy']:.2f}%")


# Plotting
names = list(results.keys())
accuracies = [results[name]["accuracy"] for name in names]
drops = [param_drops[name] for name in names]

plt.plot(drops, accuracies, marker='o')
plt.xlabel("Parameter Drop (%)")
plt.ylabel("Accuracy (%)")
plt.title("Parameter Drop vs Accuracy")
plt.grid(True)
for i, name in enumerate(names):
    plt.annotate(name, (drops[i], accuracies[i]), textcoords="offset points", xytext=(5,5), ha='center')
plt.show()