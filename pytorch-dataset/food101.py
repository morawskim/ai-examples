import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pathlib
from PIL import Image
import torchvision

import json
from helper import TinyFoodModel, train

# Setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

class Food101DataSet(Dataset):
    def __init__(self, root: str, mode: str, transform=None):
        self.root = pathlib.Path(root)
        self.transform = transform

        f = open('%s/meta/%s.json' % (root, mode))
        data = json.load(f)
        f.close()
        # self._data = data['sushi'] + data['pizza'] + data['steak']
        self._data = data['sushi'] + data['pizza']
        self.class_to_idx = {"sushi": 0, "pizza": 1, "steak": 2}
        self.classes = ["sushi", "pizza", "steak"]

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int):
        img = self._load_image(index)
        class_name = self._data[index].split('/', 1)[0]
        class_idx = self.class_to_idx[class_name]

        return self.transform(img), class_idx

    def _load_image(self, index):
        return Image.open("%s/images/%s.jpg" % (self.root, self._data[index]))

train_dataset = Food101DataSet("data/food101", mode="train", transform=torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(224, 224)),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.ToTensor(),
]))

test_dataset = Food101DataSet("data/food101", mode="test", transform=torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(224, 224)),
    torchvision.transforms.ToTensor(),
]))

train_dataloader = DataLoader(train_dataset, batch_size=24, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=24, shuffle=False)

torch.manual_seed(42)
torch.cuda.manual_seed(42)

model = TinyFoodModel(in_channels=3, hidden_units=100, out_channels=len(train_dataset.classes)).to(device)
train(model, train_dataloader, test_dataloader, nn.CrossEntropyLoss(), torch.optim.Adam(model.parameters(), lr=0.001), device)
