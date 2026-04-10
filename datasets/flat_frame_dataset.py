"""
Dataset that loads pre-extracted frames from a flat jpegs/ directory.
Frames are named {video_id}_{NNNN}.jpg and labels come from a CSV file.
"""
import csv
import os
import random

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class TwoTransform:
    """Apply two independent random augmentations to the same image."""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return self.transform(x), self.transform(x)


class FlatFrameDataset(Dataset):
    """Loads frames from a flat directory where files are {video_id}_{NNNN}.jpg."""

    def __init__(
        self,
        jpegs_dir,
        metadata_csv,
        split="train",
        num_frames=8,
        image_size=299,
        split_ratio=(0.8, 0.1, 0.1),
    ):
        self.jpegs_dir = jpegs_dir
        self.num_frames = num_frames
        self.image_size = image_size
        self.transform = self._build_transform(split)
        self.samples = self._load_samples(metadata_csv, split, split_ratio)

    def _build_transform(self, split):
        if split == "train":
            aug = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
            return TwoTransform(aug)
        else:
            base = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
            return TwoTransform(base)

    def _load_samples(self, metadata_csv, split, split_ratio):
        real_samples = []
        fake_samples = []

        with open(metadata_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                file_path = row["File Path"]
                video_id = os.path.splitext(os.path.basename(file_path))[0]
                label = 0 if row["Label"] == "REAL" else 1
                source = file_path.split("/")[0]

                # Check that at least the first frame exists
                first_frame = os.path.join(self.jpegs_dir, f"{video_id}_0000.jpg")
                if not os.path.exists(first_frame):
                    continue

                if label == 0:
                    real_samples.append((video_id, label, source))
                else:
                    fake_samples.append((video_id, label, source))

        real_samples.sort(key=lambda x: x[0])
        fake_samples.sort(key=lambda x: x[0])
        random.Random(42).shuffle(real_samples)
        random.Random(42).shuffle(fake_samples)

        def split_list(lst, ratios):
            n = len(lst)
            train_end = int(n * ratios[0])
            val_end = train_end + int(n * ratios[1])
            return lst[:train_end], lst[train_end:val_end], lst[val_end:]

        real_train, real_val, real_test = split_list(real_samples, split_ratio)
        fake_train, fake_val, fake_test = split_list(fake_samples, split_ratio)

        splits = {
            "train": real_train + fake_train,
            "val": real_val + fake_val,
            "test": real_test + fake_test,
        }

        result = splits[split]
        random.Random(42).shuffle(result)

        print(f"  {split}: {len(result)} videos "
              f"({sum(1 for s in result if s[1] == 0)} real, "
              f"{sum(1 for s in result if s[1] == 1)} fake)")

        return result

    def _load_frames(self, video_id):
        """Load frames named {video_id}_{NNNN}.jpg from the flat directory."""
        frames = []
        for i in range(self.num_frames):
            path = os.path.join(self.jpegs_dir, f"{video_id}_{i:04d}.jpg")
            if os.path.exists(path):
                img = Image.open(path).convert("RGB")
                frames.append(img)

        if len(frames) == 0:
            return None

        # Pad if fewer frames than expected
        while len(frames) < self.num_frames:
            frames.append(frames[-1])

        return frames

    def __getitem__(self, index):
        video_id, label, m_type = self.samples[index]
        frames = self._load_frames(video_id)

        if frames is None:
            zeros = torch.zeros(self.num_frames, 3, self.image_size, self.image_size)
            return zeros, zeros, label, m_type

        view1_list, view2_list = [], []
        for f in frames:
            v1, v2 = self.transform(f)
            view1_list.append(v1)
            view2_list.append(v2)

        view1 = torch.stack(view1_list)
        view2 = torch.stack(view2_list)
        return view1, view2, label, m_type

    def __len__(self):
        return len(self.samples)
