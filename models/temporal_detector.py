"""
Temporal deepfake detector: Xception backbone + cross-attention texture fusion + BiLSTM.
Combines TFFNet's per-frame consistency with temporal sequence modeling.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone import XceptionBackbone, CrossAttention


class TextureFusion(nn.Module):
    """
    Fuses texture features from two augmented views using cross-attention + gating.
    Ported from TFFNet's fusion_forward.
    """

    def __init__(self):
        super().__init__()
        # Downsample t1 (128ch, 74x74) to match t2 (256ch, 37x37) spatially
        self.t_down = nn.AvgPool2d(2, stride=2)

        # Conv to merge t1+t2 (128+256=384 → 128)
        self.t_conv_1 = nn.Sequential(
            nn.Conv2d(384, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.cross_attention = CrossAttention(in_channels=128)

        # Concat path (256 → 128)
        self.t_conv_concat = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Add path for gating (128 → 128)
        self.t_conv_add = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

    def forward(self, t1_v1, t2_v1, t1_v2, t2_v2):
        """
        Fuse texture features from two augmented views.

        Args:
            t1_v1: [N, 128, 74, 74] - block1 textures, view 1
            t2_v1: [N, 256, 37, 37] - block2 textures, view 1
            t1_v2: [N, 128, 74, 74] - block1 textures, view 2
            t2_v2: [N, 256, 37, 37] - block2 textures, view 2
        Returns:
            fused: [N, 128] - fused texture feature vector
        """
        # Downsample t1 to match t2 spatial size
        t1_v1 = self.t_down(t1_v1)  # [N, 128, 37, 37]
        t1_v2 = self.t_down(t1_v2)

        # Concatenate t1+t2 per view → [N, 384, 37, 37]
        feat_v1 = torch.cat([t1_v1, t2_v1], dim=1)
        feat_v2 = torch.cat([t1_v2, t2_v2], dim=1)

        # Reduce to 128 channels
        feat_v1 = self.t_conv_1(feat_v1)  # [N, 128, 37, 37]
        feat_v2 = self.t_conv_1(feat_v2)

        # Cross-attention: each view attends to the other
        attn_v1 = self.cross_attention(feat_v2, feat_v1, feat_v1)
        attn_v2 = self.cross_attention(feat_v1, feat_v2, feat_v2)

        # Gated fusion
        t_concat = torch.cat([attn_v1, attn_v2], dim=1)  # [N, 256, 37, 37]
        t_add = attn_v1 + attn_v2                          # [N, 128, 37, 37]

        t_concat = self.t_conv_concat(t_concat)  # [N, 128, 37, 37]
        t_add = self.t_conv_add(t_add)            # [N, 128, 37, 37]

        # Gating: sigmoid of sum controls what passes through
        fused = t_concat * torch.sigmoid(t_add)   # [N, 128, 37, 37]

        # Global average pool → [N, 128]
        fused = F.adaptive_avg_pool2d(fused, (1, 1))
        fused = fused.view(fused.size(0), -1)
        return fused


class TemporalDeepfakeDetector(nn.Module):
    """
    Per-frame: two augmented views → Xception → cross-attention texture fusion
    Temporal: fused features → BiLSTM → video-level and frame-level classification
    """

    FUSED_DIM = 128 + 256  # texture (128) + projected high-level (256)

    def __init__(
        self,
        num_classes=2,
        pretrained_backbone=True,
        freeze_backbone=False,
        lstm_hidden_dim=512,
        lstm_num_layers=2,
        lstm_dropout=0.3,
    ):
        super().__init__()
        self.backbone = XceptionBackbone(pretrained=pretrained_backbone)
        self.texture_fusion = TextureFusion()

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Project 2048-dim backbone features to 256-dim
        self.feat_proj = nn.Sequential(
            nn.Linear(XceptionBackbone.FEATURE_DIM, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )

        # Per-frame fused feature: 128 (texture) + 256 (projected) = 384
        fused_dim = self.FUSED_DIM

        self.lstm = nn.LSTM(
            input_size=fused_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0.0,
        )

        lstm_output_dim = lstm_hidden_dim * 2  # bidirectional

        # Video-level head (from final LSTM hidden state)
        self.video_head = nn.Sequential(
            nn.Linear(lstm_output_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

        # Frame-level head (from each LSTM timestep)
        self.frame_head = nn.Sequential(
            nn.Linear(lstm_output_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def _process_views(self, view1, view2):
        """
        Process two augmented views through backbone and fuse.

        Args:
            view1: [B, T, 3, H, W]
            view2: [B, T, 3, H, W]
        Returns:
            fused_seq: [B, T, 384] - per-frame fused features
            feat_v1: [B*T, 2048] - raw features view 1 (for consistency loss)
            feat_v2: [B*T, 2048] - raw features view 2 (for consistency loss)
        """
        B, T, C, H, W = view1.shape

        # Flatten to process all frames at once
        v1_flat = view1.view(B * T, C, H, W)
        v2_flat = view2.view(B * T, C, H, W)

        # Backbone forward for both views
        t1_v1, t2_v1, feat_v1 = self.backbone(v1_flat)
        t1_v2, t2_v2, feat_v2 = self.backbone(v2_flat)

        # Texture fusion via cross-attention
        texture_fused = self.texture_fusion(t1_v1, t2_v1, t1_v2, t2_v2)  # [B*T, 128]

        # Project high-level features and average both views
        proj_v1 = self.feat_proj(feat_v1)  # [B*T, 256]
        proj_v2 = self.feat_proj(feat_v2)
        proj_avg = (proj_v1 + proj_v2) / 2

        # Combine: texture + projected features
        fused = torch.cat([texture_fused, proj_avg], dim=1)  # [B*T, 384]
        fused_seq = fused.view(B, T, -1)  # [B, T, 384]

        return fused_seq, feat_v1, feat_v2

    def forward(self, view1, view2):
        """
        Args:
            view1: [B, T, 3, H, W] - first augmented view
            view2: [B, T, 3, H, W] - second augmented view
        Returns:
            video_logits: [B, num_classes]
            frame_logits: [B, T, num_classes]
            feat_v1: [B*T, 2048] - for consistency loss
            feat_v2: [B*T, 2048] - for consistency loss
        """
        fused_seq, feat_v1, feat_v2 = self._process_views(view1, view2)

        # BiLSTM over temporal sequence
        lstm_out, (h_n, _) = self.lstm(fused_seq)  # [B, T, hidden*2]

        # Video-level: concat final forward + backward hidden states
        h_forward = h_n[-2]
        h_backward = h_n[-1]
        video_repr = torch.cat([h_forward, h_backward], dim=1)
        video_logits = self.video_head(video_repr)

        # Frame-level
        frame_logits = self.frame_head(lstm_out)

        return video_logits, frame_logits, feat_v1, feat_v2
