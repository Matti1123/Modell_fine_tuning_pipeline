import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision

from scripts.dataset import PetDataset

# ---- Config ----
root = "/mnt/c/Users/flets/OneDrive/Documents/Uni allgemein/Bachelor_arbeit/erste_segmentierung/oxford_pets"
batch_size = 4
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)

# ---- Dataset ----
dataset = PetDataset(root)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ---- Model (Pretrained DeepLab als schneller Test) ----
model = torchvision.models.segmentation.deeplabv3_resnet50(weights="DEFAULT")
model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)  # 1 Klasse (Tier)
model = model.to(device)

# ---- Loss + Optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ---- Mini Training Loop ----
model.train()

for i, (images, masks) in enumerate(loader):
    if i >= 3:   # nur 3 Batches!
        break

    images = images.to(device)
    masks = masks.to(device)

    outputs = model(images)["out"]

    loss = criterion(outputs, masks)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Batch {i} - Loss: {loss.item():.4f}")

print("Sanity training finished successfully 🚀")
