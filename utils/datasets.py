import numpy as np
import cv2
import os
from torch.utils.data import Dataset
import torch

patch_size, stride = 40, 10
aug_times = 1
scales = [1, 0.9, 0.8, 0.7]
batch_size = 128
class TestDataset(Dataset):
    def __init__(self, root, sigma):
        super(TestDataset, self).__init__()
        self.root = root
        self.sigma = sigma
        names = os.listdir(self.root)
        data = []
        for name in names:
            path = os.path.join(self.root, name)
            img = cv2.imread(path, 0)
            img = np.array(img, dtype='float32') / 255.0
            data.append(img)
        self.data = data

    def __getitem__(self, item):
        img1 = self.data[item]
        img = np.expand_dims(img1, axis=0)
        img_ = torch.from_numpy(img)
        noise = torch.randn(img_.size()).mul_(self.sigma / 255.0)
        imgy = img_ + noise
        imgx = img_
        return imgy, imgx

    def __len__(self):
        return len(self.data)

class mydataset(Dataset):
    def __init__(self, root, sigma, transform=None):
        super(mydataset, self).__init__()
        self.root = root
        self.sigma = sigma
        names = os.listdir(self.root)
        data = []
        for name in names:
            img_path = os.path.join(self.root, name)
            img = cv2.imread(img_path, 0)
            patches = self.generate_patch(img)
            for patch in patches:
                data.append(patch)
        data = np.array(data, dtype='float32')/255.0
        data = np.expand_dims(data, axis=3)
        self.data = torch.from_numpy(data.transpose(0, 3, 1, 2))

    def __getitem__(self, index):
        batch_x = self.data[index]
        noise = torch.randn(batch_x.size()).mul_(self.sigma/255.0)
        batch_y = batch_x + noise
        return batch_y, batch_x

    def __len__(self):
        return self.data.size(0)

    def generate_patch(self, img):
        patches = []
        h, w = img.shape
        for s in scales:
            h_scaled, w_scaled = int(h * s), int(w * s)
            img_scaled = cv2.resize(img, (h_scaled, w_scaled), interpolation=cv2.INTER_CUBIC)
            for i in range(0, h_scaled - patch_size + 1, stride):
                for j in range(0, w_scaled - patch_size + 1, stride):
                    x = img_scaled[i:i + patch_size, j:j + patch_size]
                    x_aug = self.data_aug(x, mode=np.random.randint(0, 8))
                    patches.append(x_aug)
        return patches

    def data_aug(self, img, mode=0):
        # data augmentation
        if mode == 0:
            return img
        elif mode == 1:
            return np.flipud(img)
        elif mode == 2:
            return np.rot90(img)
        elif mode == 3:
            return np.flipud(np.rot90(img))
        elif mode == 4:
            return np.rot90(img, k=2)
        elif mode == 5:
            return np.flipud(np.rot90(img, k=2))
        elif mode == 6:
            return np.rot90(img, k=3)
        elif mode == 7:
            return np.flipud(np.rot90(img, k=3))