import torch
from torch import nn
import os
from torch.utils.data import Dataset, DataLoader
import pathlib
from PIL import Image
import torchvision

# Setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

class CustomImageDateSet(Dataset):
    def __init__(self, image_path: str, transform):
        self._image_path = image_path
        self.transform = transform
        self.paths = list(pathlib.Path(image_path).glob("*/*.jpg"))
        self.classes, self.class_to_idx = self._get_classes_and_map()

    def _get_classes_and_map(self):
        classes = sorted(entry.name for entry in os.scandir(self._image_path) if entry.is_dir())
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        return classes, class_to_idx

    def _load_image(self, index):
        return Image.open(self.paths[index])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index: int):
        img = self._load_image(index)
        class_name = pathlib.Path(self.paths[index]).parent.stem
        class_idx = self.class_to_idx[class_name]

        return self.transform(img), class_idx


class TinyFoodModel(nn.Module):
    def __init__(self, in_channels, hidden_units, out_channels):
        super().__init__()
        self.step1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.step2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.step3 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 16 * 16, out_features=out_channels),
        )
    def forward(self, x):
        x = self.step1(x)
        x = self.step2(x)
        x = self.step3(x)

        return x

def train_step(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device):
    model.train()
    train_loss, train_acc = 0, 0

    for batch, (x, y) in enumerate(dataloader):
        x,y = x.to(device), y.to(device)
        y_pred = model(x)

        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        device):
    model.eval()
    with torch.inference_mode():
        test_loss, test_acc = 0, 0
        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            y_pred = model(x)

            loss = loss_fn(y_pred, y)
            test_loss += loss.item()
            test_pred_labels = y_pred.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))

        test_loss = test_loss / len(dataloader)
        test_acc = test_acc / len(dataloader)
        return test_loss, test_acc

def train(model: torch.nn.Module, dataloader_train: torch.utils.data.DataLoader, dataloader_test: torch.utils.data.DataLoader, loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer, device):
    for epoch in range(10):
        train_loss, train_acc = train_step(model, dataloader_train, loss_fn, optimizer, device)
        test_loss, test_acc = test_step(model, dataloader_test, loss_fn, device)

        print(f"Epoch: {epoch} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")


train_dataset = CustomImageDateSet("data/pizza_steak_sushi/train", torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(64, 64)),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.ToTensor(),
]))
test_dataset = CustomImageDateSet("data/pizza_steak_sushi/test", torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(64, 64)),
    torchvision.transforms.ToTensor(),
]))

train_dataloader = DataLoader(train_dataset, batch_size=24, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=24, shuffle=False)

torch.manual_seed(42)
torch.cuda.manual_seed(42)

model = TinyFoodModel(in_channels=3, hidden_units=10, out_channels=len(train_dataset.classes)).to(device)
train(model, train_dataloader, test_dataloader, nn.CrossEntropyLoss(), torch.optim.Adam(model.parameters(), lr=0.001), device)

# img_batch, label_batch = next(iter(train_dataloader))
# img_single, label_single = img_batch[0].unsqueeze(dim=0), label_batch[0]
# print(f"Single image shape: {img_single.shape}\n")
#
# model.eval()
# with torch.inference_mode():
#     pred = model(img_single.to(device))
