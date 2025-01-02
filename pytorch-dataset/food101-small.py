import torch
from torch import nn
import os
from torch.utils.data import Dataset, DataLoader
import pathlib
from PIL import Image
import torchvision
from helper import TinyFoodModel, train

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

train_dataset = CustomImageDateSet("data/pizza_steak_sushi/train", torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(224, 224)),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.ToTensor(),
]))
test_dataset = CustomImageDateSet("data/pizza_steak_sushi/test", torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(224, 224)),
    torchvision.transforms.ToTensor(),
]))

train_dataloader = DataLoader(train_dataset, batch_size=24, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=24, shuffle=False)

torch.manual_seed(42)
torch.cuda.manual_seed(42)

model = TinyFoodModel(in_channels=3, hidden_units=100, out_channels=len(train_dataset.classes)).to(device)
train(model, train_dataloader, test_dataloader, nn.CrossEntropyLoss(), torch.optim.Adam(model.parameters(), lr=0.001), device)

# img_batch, label_batch = next(iter(train_dataloader))
# img_single, label_single = img_batch[0].unsqueeze(dim=0), label_batch[0]
# print(f"Single image shape: {img_single.shape}\n")
#
# model.eval()
# with torch.inference_mode():
#     pred = model(img_single.to(device))
