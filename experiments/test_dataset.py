from scripts.dataset import PetDataset

root = "/mnt/c/Users/flets/OneDrive/Documents/Uni allgemein/Bachelor_arbeit/erste_segmentierung/oxford_pets"

dataset = PetDataset(root)

print("Anzahl Bilder:", len(dataset))

img, mask = dataset[0]
print("Image shape:", img.shape)
print("Mask shape:", mask.shape)
print("Mask unique values:", mask.unique())
