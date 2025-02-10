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
            # nn.Sigmoid(),
            nn.ReLU(),
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

# Output Q1 - Sigmoid-RELU
# Model -  XORModel(
#   (net): Sequential(
#     (0): Linear(in_features=2, out_features=4, bias=True)
#     (1): Sigmoid()
#     (2): Linear(in_features=4, out_features=1, bias=True)
#     (3): ReLU()
#   )
# )
# Ep: 1; Loss = 0.3356306036002934
# Ep: 101; Loss = 0.2808557003736496
# Ep: 201; Loss = 0.28683094680309296
# Ep: 301; Loss = 0.2807694524526596
# Ep: 401; Loss = 0.2846277691423893
# Ep: 501; Loss = 0.28247008472681046
# Ep: 601; Loss = 0.2811063341796398
# Ep: 701; Loss = 0.27075546979904175
# Ep: 801; Loss = 0.2720358520746231
# Ep: 901; Loss = 0.26235876604914665
# Ep: 1001; Loss = 0.2508261762559414
# Ep: 1101; Loss = 0.24174083955585957
# Ep: 1201; Loss = 0.21833216398954391
# Ep: 1301; Loss = 0.19326754286885262
# Ep: 1401; Loss = 0.1521787215024233
# Ep: 1501; Loss = 0.11342826206237078
# Ep: 1601; Loss = 0.07526187924668193
# Ep: 1701; Loss = 0.04234315943904221
# Ep: 1801; Loss = 0.021715435897931457
# Ep: 1901; Loss = 0.01049206149764359
# Ep: 2001; Loss = 0.00446448934962973
# Ep: 2101; Loss = 0.0017943755665328354
# Ep: 2201; Loss = 0.000722220262105111
# Ep: 2301; Loss = 0.0002623838918225374
# Ep: 2401; Loss = 9.88814626907697e-05
# Ep: 2501; Loss = 3.747573146029026e-05
# Ep: 2601; Loss = 1.3714176930079702e-05
# Ep: 2701; Loss = 4.946423302953917e-06
# Ep: 2801; Loss = 1.8029813872999512e-06
# Ep: 2901; Loss = 6.715381744015758e-07
# Ep: 3001; Loss = 2.3396844284206963e-07
# Ep: 3101; Loss = 9.062126338221788e-08
# Ep: 3201; Loss = 3.1301418346174614e-08
# Ep: 3301; Loss = 1.155890316795194e-08
# Ep: 3401; Loss = 3.9825209796617855e-09
# Ep: 3501; Loss = 1.47857548427055e-09
# Ep: 3601; Loss = 5.622275978112157e-10
# Ep: 3701; Loss = 2.0333601469246787e-10
# Ep: 3801; Loss = 7.340617003137595e-11
# Ep: 3901; Loss = 3.019451355612546e-11
# Ep: 4001; Loss = 1.3805845355818747e-11
# Ep: 4101; Loss = 5.023537141823908e-12
# Ep: 4201; Loss = 2.4229507289419416e-12
# Ep: 4301; Loss = 9.983125437429408e-13
# Ep: 4401; Loss = 5.400124791776761e-13
# Ep: 4501; Loss = 5.719869022868806e-13
# Ep: 4601; Loss = 5.719869022868806e-13
# Ep: 4701; Loss = 5.719869022868806e-13
# Ep: 4801; Loss = 5.719869022868806e-13
# Ep: 4901; Loss = 5.719869022868806e-13
# ('net.0.weight', Parameter containing:
# tensor([[ 3.5057,  3.6116],
#         [ 0.9727,  1.1870],
#         [-1.4868, -0.4859],
#         [ 0.4181,  1.1551]], device='cuda:0', requires_grad=True))
# ('net.0.bias', Parameter containing:
# tensor([-0.8019, -1.2945,  0.8979, -0.4503], device='cuda:0',
#        requires_grad=True))
# ('net.2.weight', Parameter containing:
# tensor([[ 3.4785, -1.9894,  1.6637, -1.8190]], device='cuda:0',
#        requires_grad=True))
# ('net.2.bias', Parameter containing:
# tensor([-1.1230], device='cuda:0', requires_grad=True))
# Accuracy: 100.00%
# Example, input is -  tensor([0., 1.], device='cuda:0')
# Output is -  0.9999992847442627


# Q1 - Sigmoid-Sigmoid
# Model -  XORModel(
#   (net): Sequential(
#     (0): Linear(in_features=2, out_features=4, bias=True)
#     (1): Sigmoid()
#     (2): Linear(in_features=4, out_features=1, bias=True)
#     (3): Sigmoid()
#   )
# )
# Ep: 1; Loss = 0.2874962445348501
# Ep: 101; Loss = 0.25210192799568176
# Ep: 201; Loss = 0.2520618587732315
# Ep: 301; Loss = 0.25200215354561806
# Ep: 401; Loss = 0.2519855983555317
# Ep: 501; Loss = 0.25194334611296654
# Ep: 601; Loss = 0.25189973413944244
# Ep: 701; Loss = 0.2518301121890545
# Ep: 801; Loss = 0.251797690987587
# Ep: 901; Loss = 0.25171851366758347
# Ep: 1001; Loss = 0.2516554296016693
# Ep: 1101; Loss = 0.2516090162098408
# Ep: 1201; Loss = 0.2515087015926838
# Ep: 1301; Loss = 0.2514445520937443
# Ep: 1401; Loss = 0.2513267323374748
# Ep: 1501; Loss = 0.25122013688087463
# Ep: 1601; Loss = 0.2511214315891266
# Ep: 1701; Loss = 0.2509598061442375
# Ep: 1801; Loss = 0.2508046589791775
# Ep: 1901; Loss = 0.2506522610783577
# Ep: 2001; Loss = 0.25045283883810043
# Ep: 2101; Loss = 0.25020333006978035
# Ep: 2201; Loss = 0.24996639415621758
# Ep: 2301; Loss = 0.2496464066207409
# Ep: 2401; Loss = 0.24930912628769875
# Ep: 2501; Loss = 0.24894239753484726
# Ep: 2601; Loss = 0.2485015094280243
# Ep: 2701; Loss = 0.24797048047184944
# Ep: 2801; Loss = 0.24741443619132042
# Ep: 2901; Loss = 0.24675191566348076
# Ep: 3001; Loss = 0.2459709532558918
# Ep: 3101; Loss = 0.245133675634861
# Ep: 3201; Loss = 0.24412097036838531
# Ep: 3301; Loss = 0.24300160259008408
# Ep: 3401; Loss = 0.2417263239622116
# Ep: 3501; Loss = 0.24031635746359825
# Ep: 3601; Loss = 0.2386898286640644
# Ep: 3701; Loss = 0.23686350882053375
# Ep: 3801; Loss = 0.23478475958108902
# Ep: 3901; Loss = 0.23251327872276306
# Ep: 4001; Loss = 0.22999488934874535
# Ep: 4101; Loss = 0.22725731879472733
# Ep: 4201; Loss = 0.22429579496383667
# Ep: 4301; Loss = 0.22107002139091492
# Ep: 4401; Loss = 0.21767902374267578
# Ep: 4501; Loss = 0.21413759514689445
# Ep: 4601; Loss = 0.21042839996516705
# Ep: 4701; Loss = 0.20663385838270187
# Ep: 4801; Loss = 0.20274492353200912
# Ep: 4901; Loss = 0.19882655702531338
# ('net.0.weight', Parameter containing:
# tensor([[ 2.7817,  2.9684],
#         [ 0.2598,  0.7817],
#         [ 0.0753,  0.3058],
#         [-0.1494,  1.0597]], device='cuda:0', requires_grad=True))
# ('net.0.bias', Parameter containing:
# tensor([-0.2945, -0.5266,  0.5760,  0.4201], device='cuda:0',
#        requires_grad=True))
# ('net.2.weight', Parameter containing:
# tensor([[ 2.8228, -0.6652, -0.5589, -1.0738]], device='cuda:0',
#        requires_grad=True))
# ('net.2.bias', Parameter containing:
# tensor([-0.8186], device='cuda:0', requires_grad=True))
# Accuracy: 75.00%
# Example, input is -  tensor([0., 1.], device='cuda:0')
# Output is -  0.5442512631416321

# Q2 - RELU-RELU
# Model -  XORModel(
#   (net): Sequential(
#     (0): Linear(in_features=2, out_features=4, bias=True)
#     (1): ReLU()
#     (2): Linear(in_features=4, out_features=1, bias=True)
#     (3): ReLU()
#   )
# )
# Ep: 1; Loss = 0.4285169031281839
# Ep: 101; Loss = 0.18759107537709951
# Ep: 201; Loss = 0.14014828146900982
# Ep: 301; Loss = 0.012666614697081968
# Ep: 401; Loss = 0.00019962210558333027
# Ep: 501; Loss = 1.9736625347377412e-06
# Ep: 601; Loss = 1.9720243216170275e-08
# Ep: 701; Loss = 1.8757351227804975e-10
# Ep: 801; Loss = 2.170708057747106e-12
# Ep: 901; Loss = 5.018208071305708e-13
# Ep: 1001; Loss = 1.7763568394002505e-13
# Ep: 1101; Loss = 2.2115642650533118e-13
# Ep: 1201; Loss = 1.4654943925052066e-13
# Ep: 1301; Loss = 2.0605739337042905e-13
# Ep: 1401; Loss = 1.3944401189291966e-13
# Ep: 1501; Loss = 1.3944401189291966e-13
# Ep: 1601; Loss = 1.3944401189291966e-13
# Ep: 1701; Loss = 1.3944401189291966e-13
# Ep: 1801; Loss = 1.3944401189291966e-13
# Ep: 1901; Loss = 1.3944401189291966e-13
# Ep: 2001; Loss = 1.3944401189291966e-13
# Ep: 2101; Loss = 1.3944401189291966e-13
# Ep: 2201; Loss = 1.3944401189291966e-13
# Ep: 2301; Loss = 1.3944401189291966e-13
# Ep: 2401; Loss = 1.3944401189291966e-13
# Ep: 2501; Loss = 1.3944401189291966e-13
# Ep: 2601; Loss = 1.3944401189291966e-13
# Ep: 2701; Loss = 1.3944401189291966e-13
# Ep: 2801; Loss = 1.3944401189291966e-13
# Ep: 2901; Loss = 1.3944401189291966e-13
# Ep: 3001; Loss = 1.3944401189291966e-13
# Ep: 3101; Loss = 1.3944401189291966e-13
# Ep: 3201; Loss = 1.3944401189291966e-13
# Ep: 3301; Loss = 1.3944401189291966e-13
# Ep: 3401; Loss = 1.3944401189291966e-13
# Ep: 3501; Loss = 1.3944401189291966e-13
# Ep: 3601; Loss = 1.3944401189291966e-13
# Ep: 3701; Loss = 1.3944401189291966e-13
# Ep: 3801; Loss = 1.3944401189291966e-13
# Ep: 3901; Loss = 1.3944401189291966e-13
# Ep: 4001; Loss = 1.3944401189291966e-13
# Ep: 4101; Loss = 1.3944401189291966e-13
# Ep: 4201; Loss = 1.3944401189291966e-13
# Ep: 4301; Loss = 1.3944401189291966e-13
# Ep: 4401; Loss = 1.3944401189291966e-13
# Ep: 4501; Loss = 1.3944401189291966e-13
# Ep: 4601; Loss = 1.3944401189291966e-13
# Ep: 4701; Loss = 1.3944401189291966e-13
# Ep: 4801; Loss = 1.3944401189291966e-13
# Ep: 4901; Loss = 1.3944401189291966e-13
# ('net.0.weight', Parameter containing:
# tensor([[ 0.1176,  0.4909],
#         [-1.0885,  1.1284],
#         [-0.8043,  0.7871],
#         [-0.7002,  0.7002]], device='cuda:0', requires_grad=True))
# ('net.0.bias', Parameter containing:
# tensor([ 2.4933e-01, -3.9888e-02,  8.0427e-01,  1.8829e-08], device='cuda:0',
#        requires_grad=True))
# ('net.2.weight', Parameter containing:
# tensor([[-0.0352,  1.3054, -1.2485,  0.8270]], device='cuda:0',
#        requires_grad=True))
# ('net.2.bias', Parameter containing:
# tensor([1.0129], device='cuda:0', requires_grad=True))
# Accuracy: 100.00%
# Example, input is -  tensor([0., 1.], device='cuda:0')
# Output is -  0.9999996423721313
