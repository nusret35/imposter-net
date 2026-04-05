"""
Loss functions for temporal deepfake detection.
Combines TFFNet-style augmentation consistency with temporal modeling losses.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AugmentationConsistencyLoss(nn.Module):
    """
    TFFNet-style consistency: two augmented views of the same frame
    should produce similar features (cosine similarity → 1).
    """

    def __init__(self):
        super().__init__()

    def forward(self, feat_v1, feat_v2):
        """
        Args:
            feat_v1: [N, D] - features from augmented view 1
            feat_v2: [N, D] - features from augmented view 2
        Returns:
            loss: scalar
        """
        feat_v1 = F.normalize(feat_v1, dim=1)
        feat_v2 = F.normalize(feat_v2, dim=1)
        cos_sim = (feat_v1 * feat_v2).sum(dim=1)  # [N]
        loss = (1 - cos_sim).mean()
        return loss


class CombinedLoss(nn.Module):
    """Combined loss: video CE + frame CE + augmentation consistency."""

    def __init__(self, consistency_weight=1.0, frame_weight=0.3, real_weight=4.0):
        super().__init__()
        weight = torch.tensor([real_weight, 1.0])
        self.video_ce = nn.CrossEntropyLoss(weight=weight)
        self.frame_ce = nn.CrossEntropyLoss(weight=weight)
        self.aug_consistency = AugmentationConsistencyLoss()
        self.consistency_weight = consistency_weight
        self.frame_weight = frame_weight

    def forward(self, video_logits, frame_logits, labels, feat_v1=None, feat_v2=None):
        """
        Args:
            video_logits: [B, C] - video-level predictions
            frame_logits: [B, T, C] - frame-level predictions
            labels: [B] - video-level labels (0=real, 1=fake)
            feat_v1: [B*T, 2048] - view 1 features (for consistency)
            feat_v2: [B*T, 2048] - view 2 features (for consistency)
        Returns:
            total_loss, loss_dict
        """
        # Video-level classification loss
        loss_video = self.video_ce(video_logits, labels)

        # Frame-level classification loss
        B, T, C = frame_logits.shape
        frame_labels = labels.unsqueeze(1).expand(-1, T).reshape(-1)
        loss_frame = self.frame_ce(frame_logits.reshape(-1, C), frame_labels)

        total = loss_video + self.frame_weight * loss_frame

        loss_dict = {
            "video_ce": loss_video.item(),
            "frame_ce": loss_frame.item(),
        }

        # Augmentation consistency loss
        if feat_v1 is not None and feat_v2 is not None:
            loss_consist = self.aug_consistency(feat_v1, feat_v2)
            total = total + self.consistency_weight * loss_consist
            loss_dict["aug_consistency"] = loss_consist.item()

        loss_dict["total"] = total.item()
        return total, loss_dict
