import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

class CustomImageDataset(Dataset):
    def __init__(self, images, labels, transform):
        self.images = images  # List of images
        self.labels = labels  # List of corresponding labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]  # Get image
        label = self.labels[idx]  # Get label
        rgb_img = np.stack([image] * 3, axis=-1)
        image = Image.fromarray(rgb_img.astype(np.uint8))
        
        inputs = self.transform(
            images=image,
            return_tensors="pt",
        )
        inputs = inputs['pixel_values'].squeeze(0)

        return inputs, label
