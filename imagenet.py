import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class ImageNetDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.imgs, self.labels = self.load_data()

    def load_data(self):
        img_paths = []
        labels = []
        class_to_idx = {}
        idx = 0

        for class_dir in os.listdir(os.path.join(self.root, 'train')):
            class_path = os.path.join(self.root, 'train', class_dir)
            class_to_idx[class_dir] = idx
            idx += 1

            for img_name in os.listdir(class_path):
                img_paths.append(os.path.join(class_path, img_name))
                labels.append(class_to_idx[class_dir])

        return img_paths, labels

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

