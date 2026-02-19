import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision
from scripts.dataset import PetDataset

# =========================
# CONFIG
# =========================

root = "/mnt/c/Users/flets/OneDrive/Documents/Uni allgemein/Bachelor_arbeit/erste_segmentierung/oxford_pets"
batch_size = 4
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)

# =========================
# DATASET
# =========================

full_dataset = PetDataset(root)

# WICHTIG: gleicher Seed wie beim Training
generator = torch.Generator().manual_seed(42)

subset_size = int(0.2 * len(full_dataset))

train_subset, test_subset = random_split(
    full_dataset,
    [subset_size, len(full_dataset) - subset_size],
    generator=generator
)

print(f"Test samples (80%): {len(test_subset)}")

test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

# =========================
# MODEL LADEN
# =========================

model = torchvision.models.segmentation.deeplabv3_resnet50(weights=None)
model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)

state_dict = torch.load("deeplabv3_finetuned.pth", map_location=device)

model.load_state_dict(state_dict, strict=False)

model = model.to(device)
model.eval()

# =========================
# IoU
# =========================

def compute_iou(preds, masks, threshold=0.5):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()

    intersection = (preds * masks).sum((1, 2, 3))
    union = (preds + masks).sum((1, 2, 3)) - intersection

    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()

# =========================
# EVALUATION
# =========================

total_iou = 0
num_batches = 0

with torch.no_grad():
    for images, masks in test_loader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)["out"]
        total_iou += compute_iou(outputs, masks)
        num_batches += 1

final_iou = total_iou / num_batches

print("\n==============================")
print(f"Test IoU on 80% dataset: {final_iou:.4f}")
print("==============================")
