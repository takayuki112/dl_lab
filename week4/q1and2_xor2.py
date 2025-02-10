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

        self.net = nn.Sequential(
            nn.Linear(2, 2),
            # nn.Sigmoid(),
            nn.ReLU(),
            nn.Linear(2, 1),
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
epochs = 10000
for ep in range(epochs):
    model.train(True)
    avg_loss = train_one_ep(ep)
    loss_list.append(avg_loss)
    if ep % 1000 == 0:
        print(f"Ep: {ep}; Loss = {avg_loss}")


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

# Q1 SR
# Model -  XORModel(
#   (net): Sequential(
#     (0): Linear(in_features=2, out_features=2, bias=True)
#     (1): Sigmoid()
#     (2): Linear(in_features=2, out_features=1, bias=True)
#     (3): ReLU()
#   )
# )
# Ep: 0; Loss = 0.301132844761014
# Ep: 1000; Loss = 0.27038102224469185
# Ep: 2000; Loss = 0.27022553235292435
# Ep: 3000; Loss = 0.26847536861896515
# Ep: 4000; Loss = 0.26516806334257126
# Ep: 5000; Loss = 0.22409657016396523
# Ep: 6000; Loss = 0.0001402673997290549
# Ep: 7000; Loss = 1.2700951401711791e-11
# Ep: 8000; Loss = 3.2152058793144533e-12
# Ep: 9000; Loss = 3.0020430585864233e-12
# ('net.0.weight', Parameter containing:
# tensor([[-1.9767,  2.0750],
#         [-3.0809,  3.3435]], device='cuda:0', requires_grad=True))
# ('net.0.bias', Parameter containing:
# tensor([ 0.7012, -2.5002], device='cuda:0', requires_grad=True))
# ('net.2.weight', Parameter containing:
# tensor([[-2.6653,  2.7713]], device='cuda:0', requires_grad=True))
# ('net.2.bias', Parameter containing:
# tensor([1.5715], device='cuda:0', requires_grad=True))
# Accuracy: 100.00%
# Example, input is -  tensor([0., 1.], device='cuda:0')
# Output is -  0.9999991655349731

# Q2 RR
# Model -  XORModel(
#   (net): Sequential(
#     (0): Linear(in_features=2, out_features=2, bias=True)
#     (1): ReLU()
#     (2): Linear(in_features=2, out_features=1, bias=True)
#     (3): ReLU()
#   )
# )
# Ep: 0; Loss = 0.28828215412795544
# Ep: 1000; Loss = 0.1324781192233786
# Ep: 2000; Loss = 0.13274903275851102
# Ep: 3000; Loss = 0.13273529073421741
# Ep: 4000; Loss = 0.13273696706164628
# Ep: 5000; Loss = 0.13273241860179041
# Ep: 6000; Loss = 0.13277134528470924
# Ep: 7000; Loss = 0.1327249385152669
# Ep: 8000; Loss = 0.13273597316810992
# Ep: 9000; Loss = 0.13247891812352464
# ('net.0.weight', Parameter containing:
# tensor([[ 0.6216,  0.6761],
#         [-0.6845,  0.6890]], device='cuda:0', requires_grad=True))
# ('net.0.bias', Parameter containing:
# tensor([-0.6318, -0.0135], device='cuda:0', requires_grad=True))
# ('net.2.weight', Parameter containing:
# tensor([[-0.8123,  0.7879]], device='cuda:0', requires_grad=True))
# ('net.2.bias', Parameter containing:
# tensor([0.4957], device='cuda:0', requires_grad=True))
# Accuracy: 75.00%
# Example, input is -  tensor([0., 1.], device='cuda:0')
# Output is -  0.9919580817222595