# U-Net Fine Tuning on Oxford-IIIT Pet Dataset

This project performs fine tuning of a U-Net architecture for image segmentation using the Oxford-IIIT Pet Dataset.

## Objective

The goal is to adapt a pretrained encoder within a U-Net model to the task of binary pet segmentation (pet vs. background).

## Approach

- Pretrained encoder (ImageNet weights)
- Decoder trained for segmentation
- CPU-based training in a Linux (WSL) environment

## Dataset

Oxford-IIIT Pet Dataset  
Task: Binary image segmentation


## Testing

- ### Epoch 1/5
- Train Loss: 0.2746
- Val Loss:   0.1947
- Val IoU:    0.7726

- ### Epoch 2/5
- Train Loss: 0.1720
- Val Loss:   0.1690
- Val IoU:    0.7750

- ### Epoch 3/5
- Train Loss: 0.1383
-  Loss:   0.1586
- Val IoU:    0.7923


- #### Epoch 4/5
- Train Loss: 0.1194
- Val Loss:   0.1783
- Val IoU:    0.7760

100%|███████████████████████████████████████████████████████████████████████████████████████| 296/296 [18:10<00:00,  3.69s/it]
