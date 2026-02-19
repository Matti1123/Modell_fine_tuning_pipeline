import matplotlib.pyplot as plt
from scripts.dataset import PetDataset

# Dataset-Pfad
root = "/mnt/c/Users/flets/OneDrive/Documents/Uni allgemein/Bachelor_arbeit/erste_segmentierung/oxford_pets"

# Dataset laden
dataset = PetDataset(root)

# Ein Beispiel auswählen
img, mask = dataset[8]  # erstes Bild

# mask ist Tensor [1, H, W] → in 2D umwandeln
mask_2d = mask[0].numpy()

# Schwarz/Weiß: nur Tier
plt.figure(figsize=(5,5))
plt.imshow(mask_2d, cmap="gray")
plt.title("Tier-Maske")
plt.savefig("tier_maske.png")

# Optional: Original Maskenwerte farbig darstellen
plt.figure(figsize=(5,5))
plt.imshow(mask[0].numpy()*2, cmap="tab10")  # Werte 0 oder 1, für Farbkarte ggf multiplizieren
plt.title("Maskenwerte")
plt.savefig("maskenwerte.png")
