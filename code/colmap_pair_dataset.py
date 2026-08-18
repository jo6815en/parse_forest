import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class ColmapPairDataset(Dataset):
    def __init__(self, dataset_dir, transform=None, return_names=False):
        """
        dataset_dir ska innehålla:
            images/
            relative_poses.json

        Returnerar:
            image_a: Tensor [3,H,W]
            image_b: Tensor [3,H,W]
            T_ab:    Tensor [4,4]

        T_ab mappar punkter från kamera/frame a till kamera/frame b.
        """
        self.dataset_dir = Path(dataset_dir)
        self.image_dir = self.dataset_dir / "images"
        self.return_names = return_names

        with open(self.dataset_dir / "relative_poses.json", "r") as f:
            self.pairs = json.load(f)

        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs[idx]

        img_a_path = self.image_dir / item["from_image"]
        img_b_path = self.image_dir / item["to_image"]

        image_a = Image.open(img_a_path).convert("RGB")
        image_b = Image.open(img_b_path).convert("RGB")

        image_a = self.transform(image_a)
        image_b = self.transform(image_b)

        T_ab = torch.tensor(item["T_ab"], dtype=torch.float32)

        sample = {
            "image_a": image_a,
            "image_b": image_b,
            "T_ab": T_ab,
        }

        if self.return_names:
            sample["image_a_name"] = item["from_image"]
            sample["image_b_name"] = item["to_image"]

        return sample