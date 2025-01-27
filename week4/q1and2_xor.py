import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt

X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype = torch.float32)
Y = torch.tensor([0.0, 1.0, 1.0, 0.0], dtype = torch.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)

class MyDataset(Dataset):
    def __init__(self, x, y):
        self.X = x
        self.Y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx].to(device), self.Y[idx].to(device)

class XORModel(nn.Module):
    def __init__(self):
        super(XORModel, self).__init__()

        # self.l1 = nn.Linear(2, 2)
        # self.a1 = nn.Sigmoid()
        #
        # self.l2 = nn.Linear(2, 1)
        # self.a2 = nn.ReLU()

        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Sigmoid(),
            # nn.ReLU(),
            nn.Linear(4, 1),
            # nn.Sigmoid()
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)



batch_size = 1

full_dataset = MyDataset(X, Y)
loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

model = XORModel().to(device)
print("Model - ", model)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.03)


def train_one_ep(ep_idx):
    total_loss = 0
    for i, data in enumerate(loader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs.flatten(), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader) * batch_size


loss_list = []
epochs = 5000
for ep in range(epochs):
    model.train(True)
    avg_loss = train_one_ep(ep)
    loss_list.append(avg_loss)
    if ep % 100 == 0:
        print(f"Ep: {ep+1}; Loss = {avg_loss}")


for param in model.named_parameters():
    print(param)

input = torch.tensor([0, 1], dtype = torch.float32).to(device)

model.eval()


def calculate_accuracy(model, loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs).flatten()
            predicted = (outputs >= 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = 100 * correct / total
    return accuracy

# Calculate accuracy
accuracy = calculate_accuracy(model, loader)
print(f"Accuracy: {accuracy:.2f}%")

print("Example, input is - ", input)
print("Output is - ", model(input).item())

plt.plot(loss_list)
plt.grid()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()




