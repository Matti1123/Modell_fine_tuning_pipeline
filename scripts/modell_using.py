import torch
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
import matplotlib.pyplot as plt
from scripts.dataset import PetDataset
from pathlib import Path
import numpy as np

# ------------------------
# 1️⃣ Pfade & Device
# ------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
model_path = BASE_DIR / "deeplabv3_finetuned.pth"

root_dir = "/mnt/c/Users/flets/OneDrive/Documents/Uni allgemein/Bachelor_arbeit/erste_segmentierung/oxford_pets"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------
# 2️⃣ Dataset + DataLoader
# ------------------------
dataset = PetDataset(root_dir)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

# ------------------------
# 3️⃣ Modell laden
# ------------------------
model = deeplabv3_resnet50(num_classes=1)
model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
model.to(device)
model.eval()

print("Modell geladen!")

# ------------------------
# 4️⃣ 10 Vorhersagen mit 3-Spalten-Ansicht
# ------------------------
count = 0
with torch.no_grad():
    for img, _ in loader:
        img = img.to(device)
        output = model(img)['out']  # [1,1,H,W]

        # 1-Klassen Vorhersage
        pred_mask = (output[0,0] > 0.3).float().cpu()  # 0/1

        img_show = img[0].cpu().permute(1,2,0)  # CHW -> HWC

        # ------------------------
        # Overlay erstellen
        # ------------------------
        overlay = img_show.clone().numpy()
        mask_np = pred_mask.numpy()
        overlay[...,0] = np.where(mask_np==1, 1.0, overlay[...,0])  # Rot verstärken
        overlay[...,1] = np.where(mask_np==1, 0.3*overlay[...,1], overlay[...,1])
        overlay[...,2] = np.where(mask_np==1, 0.3*overlay[...,2], overlay[...,2])

        # ------------------------
        # Plot: Original | Maske | Overlay
        # ------------------------
        plt.figure(figsize=(15,5))

        plt.subplot(1,3,1)
        plt.title("Original Image")
        plt.imshow(img_show)
        plt.axis("off")

        plt.subplot(1,3,2)
        plt.title("Predicted Mask (B/W)")
        plt.imshow(pred_mask, cmap="gray")
        plt.axis("off")

        plt.subplot(1,3,3)
        plt.title("Overlay (Red Mask)")
        plt.imshow(overlay)
        plt.axis("off")

        plt.savefig(f"prediction_overlay_{count}.png")
        plt.close()

        count += 1
        if count >= 10:
            break

print("10 Overlay-Predictions (Original | Mask B/W | Overlay) gespeichert!")
