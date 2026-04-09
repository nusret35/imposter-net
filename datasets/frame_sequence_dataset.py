"""
Dataset that loads pre-extracted frames from disk instead of decoding video.
Much faster data loading — eliminates the video decoding bottleneck.
"""
import csv
import os
import random
from pathlib import Path

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


class FrameSequenceDataset(Dataset):
    """Loads pre-extracted frames (from extract_frames.py) instead of videos."""

    def __init__(
        self,
        frames_dir,
        split="train",
        num_frames=16,
        image_size=299,
        split_ratio=(0.8, 0.1, 0.1),
    ):
        """
        Args:
            frames_dir: Path to directory with extracted frames and metadata.csv
            split: One of 'train', 'val', 'test'.
            num_frames: Number of frames per video.
            image_size: Resize frames to this size.
            split_ratio: (train, val, test) ratio.
        """
        self.frames_dir = frames_dir
        self.num_frames = num_frames
        self.image_size = image_size
        self.transform = self._build_transform(split)
        self.samples = self._load_samples(split, split_ratio)

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

    def _load_samples(self, split, split_ratio):
        """Load metadata and do stratified split."""
        real_samples = []
        fake_samples = []

        meta_path = os.path.join(self.frames_dir, "metadata.csv")
        with open(meta_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                video_id = row["video_id"]
                label = 0 if row["label"] == "REAL" else 1
                source = row["source"]
                frame_dir = os.path.join(self.frames_dir, video_id)
                if not os.path.isdir(frame_dir):
                    continue
                if label == 0:
                    real_samples.append((frame_dir, label, source))
                else:
                    fake_samples.append((frame_dir, label, source))

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

    def _load_frames(self, frame_dir):
        """Load pre-extracted frame images from a directory."""
        frame_files = sorted(f for f in os.listdir(frame_dir) if f.endswith(".jpg"))

        if len(frame_files) == 0:
            return None

        # Take evenly spaced frames if more than needed
        if len(frame_files) >= self.num_frames:
            indices = [int(i * len(frame_files) / self.num_frames) for i in range(self.num_frames)]
            frame_files = [frame_files[i] for i in indices]

        frames = []
        for fname in frame_files:
            img = Image.open(os.path.join(frame_dir, fname)).convert("RGB")
            frames.append(img)

        # Pad if needed
        while len(frames) < self.num_frames:
            frames.append(frames[-1])

        return frames

    def __getitem__(self, index):
        frame_dir, label, m_type = self.samples[index]
        frames = self._load_frames(frame_dir)

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
