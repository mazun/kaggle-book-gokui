from typing import Literal
import numpy as np
import os
import sklearn.model_selection
import torch
import torch.utils.data
import torchvision

from pathlib import Path
from torch import nn
from torchvision import transforms
from tqdm import tqdm

device: str = "cuda" if torch.cuda.is_available() else "cpu"


def setup_train_val_split(labels, dryrun=False, seed=0):
    x = np.arange(len(labels))
    y = np.array(labels)
    splitter = sklearn.model_selection.StratifiedShuffleSplit(
        n_splits=1, train_size=0.8, random_state=seed
    )
    train_indices, val_indices = next(splitter.split(x, y))

    if dryrun:
        train_indices = np.random.choice(train_indices, 100, replace=False)
        val_indices = np.random.choice(val_indices, 100, replace=False)

    return train_indices, val_indices


def setup_center_crop_transform():
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def get_labels(dataset: torchvision.datasets.DatasetFolder):
    return [sample[1] for sample in dataset.samples]


def setup_train_val_datasets(data_dir: Path, dryrun=False):
    dataset = torchvision.datasets.ImageFolder(
        data_dir / "train", transform=setup_center_crop_transform()
    )
    labels = get_labels(dataset)
    train_indices, val_indices = setup_train_val_split(labels, dryrun)

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    return train_dataset, val_dataset


def setup_train_val_loaders(data_dir: Path, batch_size, dryrun=False):
    train_dataset, val_dataset = setup_train_val_datasets(data_dir, dryrun)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8,
    )
    val_loder = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, num_workers=8
    )
    return train_loader, val_loder


def train_1epoch(
    model: torchvision.models.ResNet,
    train_loader: torch.utils.data.DataLoader,
    lossfun: nn.CrossEntropyLoss,
    optimizer: torch.optim.Optimizer,
    device: str,
):
    model.train()
    total_loss, total_acc = 0.0, 0.0

    for x, y in tqdm(train_loader):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = lossfun(out, y)
        _, pred = torch.max(out.detach(), 1)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_acc += torch.sum(pred == y)

    avg_loss = total_loss / len(train_loader.dataset)
    avg_acc = total_acc / len(train_loader.dataset)
    return avg_acc, avg_loss


def validate_1epoch(
    model: torchvision.models.ResNet,
    val_loader: torch.utils.data.DataLoader,
    lossfun: nn.CrossEntropyLoss,
    device: str,
):
    model.eval()
    total_loss, total_acc = 0.0, 0.0

    with torch.no_grad():
        for x, y in tqdm(val_loader):
            x = x.to(device)
            y = y.to(device)

            out = model(x)
            loss = lossfun(out.detach(), y)
            _, pred = torch.max(out, 1)
            total_loss += loss.item() * x.size(0)
            total_acc += torch.sum(pred == y)

    avg_loss = total_loss / len(val_loader.dataset)
    avg_acc = total_acc / len(val_loader.dataset)
    return avg_acc, avg_loss


def train(
    model: torchvision.models.ResNet,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    n_epochs: int,
    device: str,
):
    lossfun = torch.nn.CrossEntropyLoss()
    for epoch in tqdm(range(n_epochs)):
        train_acc, train_loss = train_1epoch(
            model, train_loader, lossfun, optimizer, device
        )
        val_acc, val_loss = validate_1epoch(model, val_loader, lossfun, device)
        print(
            f"epoch={epoch}, train loss={train_loss}, train accuracy={train_acc}, val loss={val_loss}, val accuracy={val_acc}"
        )


def train_subsec5(
    data_dir: Path,
    batch_size: int,
    dryrun=False,
    device: str = "cuda",
):
    model = torchvision.models.resnet50(
        weights=torchvision.models.resnet.ResNet50_Weights.IMAGENET1K_V1
    )
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    train_loader, val_loader = setup_train_val_loaders(data_dir, batch_size, dryrun)
    train(model, optimizer, train_loader, val_loader, n_epochs=1, device=device)
    return model


def setup_test_loader(data_dir, batch_size, dryrun):
    dataset = torchvision.datasets.ImageFolder(
        data_dir / "test", transform=setup_center_crop_transform()
    )
    image_ids = [
        os.path.splitext(os.path.basename(path))[0] for path, _ in dataset.imgs
    ]
    if dryrun:
        dataset = torch.utils.data.Subset(dataset, range(0, 100))
        image_ids = image_ids[:100]
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=8)
    return loader, image_ids


def predict(
    model: torchvision.models.ResNet, loader: torch.utils.data.DataLoader, device: str
):
    pred_fun = torch.nn.Softmax(dim=1)
    preds = []
    for x, _ in tqdm(loader):
        with torch.set_grad_enabled(False):
            x = x.to(device)
            y = pred_fun(model(x))
        y = y.cpu().numpy()
        y = y[:, 1]
        preds.append(y)
    preds = np.concatenate(preds)
    return preds


def write_prediction(
    image_ids: [str], prediction: [int], clip_threashold: float, out_path: Path
):
    with open(out_path, "w") as f:
        f.write("id,label\n")
        for i, p in zip(image_ids, prediction):
            p = np.clip(p, clip_threashold, 1.0 - clip_threashold)
            f.write(f"{i},{p}\n")


def predict_subsec5(
    model: torchvision.models.ResNet,
    data_dir: Path,
    out_dir: Path,
    batch_size: int,
    clip_threashold: float,
    dryrun=False,
    device="cuda",
):
    test_loader, image_ids = setup_test_loader(data_dir, batch_size, dryrun=dryrun)
    preds = predict(model, test_loader, device)
    write_prediction(image_ids, preds, clip_threashold, out_dir / "out.csv")


dryrun = False
batch_size = 50
clip_threashold = 0.0125
model = train_subsec5(Path("data"), batch_size, dryrun=dryrun, device=device)
predict_subsec5(
    model, Path("data"), Path("."), batch_size, clip_threashold, dryrun=dryrun, device=device
)
