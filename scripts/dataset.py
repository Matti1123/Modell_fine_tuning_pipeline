import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import torch


class PetDataset(Dataset):
    def __init__(self, root_dir):
        self.img_dir = os.path.join(root_dir, "images")
        self.mask_dir = os.path.join(root_dir, "annotations/trimaps")

        # Nur JPG-Dateien verwenden
        self.files = [f for f in os.listdir(self.img_dir) if f.endswith(".jpg")]

        self.img_transform = T.Compose([
            T.Resize((128, 128)),
            T.ToTensor()
        ])

        self.mask_transform = T.Compose([
            T.Resize((128, 128)),
            T.PILToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx]
        mask_name = img_name.replace(".jpg", ".png")

        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        image = self.img_transform(image)
        mask = self.mask_transform(mask).float()

        # Oxford Trimap: 1 = Pet, 2 = Background, 3 = Border
        mask = (mask == 1).float()

        return image, mask
