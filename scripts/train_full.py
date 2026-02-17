import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision
from tqdm import tqdm


from scripts.dataset import PetDataset


# =========================
# CONFIG
# =========================

root = "/mnt/c/Users/flets/OneDrive/Documents/Uni allgemein/Bachelor_arbeit/erste_segmentierung/oxford_pets"
batch_size = 4
learning_rate = 1e-4
num_epochs = 5
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)


# =========================
# DATASET
# =========================

dataset = PetDataset(root)

# --- Nur 20% des gesamten Datasets nutzen ---
subset_size = int(0.2 * len(dataset))
dataset, _ = random_split(dataset, [subset_size, len(dataset) - subset_size])

# --- Danach 80/20 Train/Val Split innerhalb der 20% ---
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")


# =========================
# MODEL (Full Fine-Tuning)
# =========================

model = torchvision.models.segmentation.deeplabv3_resnet50(weights="DEFAULT")

# Output auf 1 Klasse ändern (binary segmentation)
model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)

model = model.to(device)


# =========================
# LOSS & OPTIMIZER
# =========================

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# =========================
# IoU METRIC
# =========================

def compute_iou(preds, masks, threshold=0.5):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()

    intersection = (preds * masks).sum((1, 2, 3))
    union = (preds + masks).sum((1, 2, 3)) - intersection

    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()


# =========================
# TRAINING LOOP
# =========================



for epoch in range(num_epochs):

    # -------- TRAINING --------
    model.train()
    train_loss = 0

    for images, masks in tqdm(train_loader):
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)["out"]
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)


    # -------- VALIDATION --------
    model.eval()
    val_loss = 0
    val_iou = 0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)["out"]
            loss = criterion(outputs, masks)

            val_loss += loss.item()
            val_iou += compute_iou(outputs, masks)

    val_loss /= len(val_loader)
    val_iou /= len(val_loader)


    # -------- LOGGING --------
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss:   {val_loss:.4f}")
    print(f"Val IoU:    {val_iou:.4f}")
    print("-" * 40)


# =========================
# SAVE MODEL
# =========================

torch.save(model.state_dict(), "deeplabv3_finetuned.pth")
print("Model saved as deeplabv3_finetuned.pth")
