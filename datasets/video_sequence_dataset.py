import csv
import os
from pathlib import Path

import cv2
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


class VideoSequenceDataset(Dataset):
    """Dataset that samples T sequential frames from FF++ videos for temporal modeling."""

    MANIPULATION_TYPES = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]

    def __init__(
        self,
        root_dir,
        split="train",
        num_frames=16,
        image_size=299,
        manipulation_types=None,
        split_ratio=(0.8, 0.1, 0.1),
    ):
        """
        Args:
            root_dir: Path to FaceForensics++_C23 directory.
            split: One of 'train', 'val', 'test'.
            num_frames: Number of frames to sample per video.
            image_size: Resize frames to this size.
            manipulation_types: List of fake types to include. None = all.
            split_ratio: (train, val, test) ratio.
        """
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.image_size = image_size
        self.split = split

        if manipulation_types is None:
            manipulation_types = self.MANIPULATION_TYPES
        self.manipulation_types = manipulation_types

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
            # At eval time, both views are identical (no randomness)
            return TwoTransform(base)

    def _load_samples(self, split, split_ratio):
        """Load video paths and labels from CSV files, then do stratified split."""
        import random

        real_samples = []
        fake_samples = []

        # Load real videos
        csv_path = os.path.join(self.root_dir, "csv", "original.csv")
        real_samples += self._parse_csv(csv_path)

        # Load fake videos for each manipulation type
        for m_type in self.manipulation_types:
            csv_path = os.path.join(self.root_dir, "csv", f"{m_type}.csv")
            if os.path.exists(csv_path):
                fake_samples += self._parse_csv(csv_path)

        # Sort each group for reproducibility, then shuffle with fixed seed
        real_samples.sort(key=lambda x: x[0])
        fake_samples.sort(key=lambda x: x[0])
        random.Random(42).shuffle(real_samples)
        random.Random(42).shuffle(fake_samples)

        # Stratified split: same ratio of real/fake in each split
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

    def _parse_csv(self, csv_path):
        """Parse FF++ CSV and return list of (video_path, label, manipulation_type)."""
        samples = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                video_path = os.path.join(self.root_dir, row["File Path"])
                label = 0 if row["Label"] == "REAL" else 1
                m_type = Path(row["File Path"]).parts[0]
                samples.append((video_path, label, m_type))
        return samples

    def _sample_frames(self, video_path):
        """Sample num_frames evenly-spaced frames from a video."""
        capture = cv2.VideoCapture(video_path)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            capture.release()
            return None

        # Evenly spaced indices
        if total_frames >= self.num_frames:
            indices = [int(i * total_frames / self.num_frames) for i in range(self.num_frames)]
        else:
            # If video is shorter than num_frames, sample all and pad by repeating last
            indices = list(range(total_frames))

        frames = []
        for idx in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, idx)
            success, frame = capture.read()
            if success:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frames.append(frame)
        capture.release()

        if len(frames) == 0:
            return None

        # Pad if needed
        while len(frames) < self.num_frames:
            frames.append(frames[-1])

        return frames

    def __getitem__(self, index):
        video_path, label, m_type = self.samples[index]
        frames = self._sample_frames(video_path)

        if frames is None:
            # Return zeros as fallback for corrupted videos
            zeros = torch.zeros(self.num_frames, 3, self.image_size, self.image_size)
            return zeros, zeros, label, m_type

        # Apply TwoTransform to each frame → two augmented views
        view1_list, view2_list = [], []
        for f in frames:
            v1, v2 = self.transform(f)
            view1_list.append(v1)
            view2_list.append(v2)

        view1 = torch.stack(view1_list)  # [T, 3, H, W]
        view2 = torch.stack(view2_list)  # [T, 3, H, W]
        return view1, view2, label, m_type

    def __len__(self):
        return len(self.samples)
