from sklearn.datasets import load_digits
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

digits = load_digits()
X = torch.tensor(digits.data, dtype=torch.float32)
y = torch.tensor(digits.target, dtype=torch.int64)
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

model = nn.Sequential(
    nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 10)
)
model.to(device)
model.train()
lossfun = nn.CrossEntropyLoss()
lossfun.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01)


def train_1epoch(
    model: nn.modules.Module,
    train_loader: DataLoader,
    lossfun: nn.modules.Module,
    optimizer: optim.Optimizer,
    device: str,
):
    model.train()
    total_loss, total_acc = 0.0, 0.0
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = lossfun(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        _, pred = torch.max(out, 1)
        total_acc += torch.sum(pred == y.data)
    avg_loss = total_loss / len(train_loader.dataset)
    avg_acc = total_acc / len(train_loader.dataset)
    return avg_acc, avg_loss


losses = []
for ep in tqdm(range(1000)):
    avg_acc, avg_loss = train_1epoch(model, dataloader, lossfun, optimizer, device)
    losses.append(avg_loss * len(dataloader.dataset))

plt.plot(losses)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()
