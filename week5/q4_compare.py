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
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])  # Added normalization

mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)  # Shuffle False for consistent evaluation

# Define the base CNN model
class BaseCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # Reduced filters, added padding
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Reduced filters, added padding
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(start_dim=1)
        )
        self.classification_head = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),  # Adjusted input size
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        features = self.net(x)
        return self.classification_head(features)

# Function to calculate the number of parameters in a model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

# Function to train and evaluate the model
def train_evaluate(model, train_loader, test_loader, learning_rate, epochs, device):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate on the test set after each epoch
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            print(f'Epoch [{epoch+1}/{epochs}], Accuracy: {accuracy:.2f}%')
        model.train()  # Put model back in train mode

    return accuracy

# Modified CNN architectures with varying parameter reductions
class ModifiedCNN1(nn.Module): #Smaller filters in conv layers
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # Reduced filters
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # Reduced filters
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(start_dim=1)
        )
        self.classification_head = nn.Sequential(
            nn.Linear(32 * 7 * 7, 64),  # Adjusted input and output size
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        features = self.net(x)
        return self.classification_head(features)

class ModifiedCNN2(nn.Module): #Smaller linear layers
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(start_dim=1)
        )
        self.classification_head = nn.Sequential(
            nn.Linear(64 * 7 * 7, 64),  # Adjusted output size
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        features = self.net(x)
        return self.classification_head(features)


class ModifiedCNN3(nn.Module):  # Fewer convolutional layers, smaller linear layers
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), # Reduced filters, removed later
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(start_dim=1)
        )
        self.classification_head = nn.Sequential(
            nn.Linear(16 * 14 * 14, 32),  # Adjusted input and output size, removed later
            nn.ReLU(),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        features = self.net(x)
        return self.classification_head(features)



# Train and evaluate models
base_model = BaseCNN().to(device)
modified_model1 = ModifiedCNN1().to(device)
modified_model2 = ModifiedCNN2().to(device)
modified_model3 = ModifiedCNN3().to(device)

base_params = count_parameters(base_model)
modified_params1 = count_parameters(modified_model1)
modified_params2 = count_parameters(modified_model2)
modified_params3 = count_parameters(modified_model3)

base_accuracy = train_evaluate(base_model, train_loader, test_loader, learning_rate, epochs, device)
modified_accuracy1 = train_evaluate(modified_model1, train_loader, test_loader, learning_rate, epochs, device)
modified_accuracy2 = train_evaluate(modified_model2, train_loader, test_loader, learning_rate, epochs, device)
modified_accuracy3 = train_evaluate(modified_model3, train_loader, test_loader, learning_rate, epochs, device)


# Calculate parameter drop and accuracy change
param_drop1 = (1 - (modified_params1 / base_params)) * 100
param_drop2 = (1 - (modified_params2 / base_params)) * 100
param_drop3 = (1 - (modified_params3 / base_params)) * 100


print(f"Base Model Parameters: {base_params}")
print(f"Modified Model 1 Parameters: {modified_params1}, Parameter Drop: {param_drop1:.2f}%, Accuracy: {modified_accuracy1:.2f}%")
print(f"Modified Model 2 Parameters: {modified_params2}, Parameter Drop: {param_drop2:.2f}%, Accuracy: {modified_accuracy2:.2f}%")
print(f"Modified Model 3 Parameters: {modified_params3}, Parameter Drop: {param_drop3:.2f}%, Accuracy: {modified_accuracy3:.2f}%")


# Plot parameter drop vs accuracy
param_drops = [0, param_drop1, param_drop2, param_drop3]  # Include base model with 0 drop
accuracies = [base_accuracy, modified_accuracy1, modified_accuracy2, modified_accuracy3]
model_names = ["Base", "Mod1", "Mod2", "Mod3"]

plt.plot(param_drops, accuracies, marker='o')
plt.xlabel("Percentage Drop in Parameters")
plt.ylabel("Accuracy (%)")
plt.title("Parameter Drop vs Accuracy")
plt.grid(True)

# Annotate the points with model names
for i, txt in enumerate(model_names):
    plt.annotate(txt, (param_drops[i], accuracies[i]), textcoords="offset points", xytext=(5,5), ha='center')

plt.show()