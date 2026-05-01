import os
import cv2
import torch
from torch.utils.data import Dataset

class PairedContourDataset(Dataset):
    def __init__(self, image_dir, img_size=128):
        self.image_paths = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        self.img_size = img_size

    def extract_contours(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.Canny(gray, 100, 200)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])

        hr = cv2.resize(img, (self.img_size, self.img_size))

        lr = cv2.resize(hr, (self.img_size // 2, self.img_size // 2))
        lr = cv2.resize(lr, (self.img_size, self.img_size))

        contour = self.extract_contours(lr)

        hr = torch.tensor(hr).permute(2, 0, 1).float() / 255
        lr = torch.tensor(lr).permute(2, 0, 1).float() / 255
        contour = torch.tensor(contour).unsqueeze(0).float() / 255

        return {"lr": lr, "contour": contour, "hr": hr}
