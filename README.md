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

Epoch 1/5
Train Loss: 0.2746
Val Loss:   0.1947
Val IoU:    0.7726
----------------------------------------

