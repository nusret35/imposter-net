import argparse
import os

import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from datasets.video_sequence_dataset import VideoSequenceDataset
from datasets.frame_sequence_dataset import FrameSequenceDataset
from datasets.flat_frame_dataset import FlatFrameDataset
from models.temporal_detector import TemporalDeepfakeDetector
from training.losses import CombinedLoss
from training.trainer import train_epoch, evaluate

load_dotenv()

DATASET_ROOT = os.getenv("DATASET_ROOT", "")


def parse_args():
    parser = argparse.ArgumentParser(description="Temporal Deepfake Detection")
    parser.add_argument("--root-dir", default=DATASET_ROOT, help="path to FF++ dataset")
    parser.add_argument("--frames-dir", default=None, help="path to pre-extracted frames (from extract_frames.py)")
    parser.add_argument("--jpegs-dir", default=None, help="path to flat jpegs directory (from extract_images.py)")
    parser.add_argument("--metadata-csv", default=None, help="path to FF++_Metadata_Shuffled.csv")
    parser.add_argument("--num-frames", type=int, default=16, help="frames to sample per video")
    parser.add_argument("--image-size", type=int, default=299, help="input image size")
    parser.add_argument("--batch-size", type=int, default=4, help="batch size")
    parser.add_argument("--epochs", type=int, default=20, help="number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--freeze-backbone", action="store_true", help="freeze Xception weights")
    parser.add_argument("--consistency-weight", type=float, default=1.0, help="augmentation consistency loss weight")
    parser.add_argument("--num-workers", type=int, default=0, help="dataloader workers")
    parser.add_argument("--device", default="cpu", help="device (cpu/cuda/mps)")
    parser.add_argument("--accum-steps", type=int, default=1, help="gradient accumulation steps (effective batch = batch-size * accum-steps)")
    parser.add_argument("--save-dir", default="checkpoints", help="directory to save model checkpoints")
    return parser.parse_args()


def make_dataset(args, split):
    if args.jpegs_dir and args.metadata_csv:
        return FlatFrameDataset(
            jpegs_dir=args.jpegs_dir, metadata_csv=args.metadata_csv,
            split=split, num_frames=args.num_frames, image_size=args.image_size,
        )
    elif args.frames_dir:
        return FrameSequenceDataset(
            frames_dir=args.frames_dir, split=split,
            num_frames=args.num_frames, image_size=args.image_size,
        )
    else:
        return VideoSequenceDataset(
            root_dir=args.root_dir, split=split,
            num_frames=args.num_frames, image_size=args.image_size,
        )


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    print("Loading datasets...")
    train_dataset = make_dataset(args, "train")
    val_dataset = make_dataset(args, "val")
    test_dataset = make_dataset(args, "test")

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print("Building model...")
    model = TemporalDeepfakeDetector(
        num_classes=2,
        pretrained_backbone=True,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    criterion = CombinedLoss(consistency_weight=args.consistency_weight).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_auc = 0.0
    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")

        train_losses, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, accum_steps=args.accum_steps)
        print(f"Train - Loss: {train_losses['total']:.4f}, Acc: {train_acc:.4f}")

        val_metrics = evaluate(model, val_loader, criterion, device)
        print(f"Val   - Acc: {val_metrics['accuracy']:.4f}, AUC: {val_metrics['auc']:.4f}")

        for m_type, m_metrics in val_metrics.get("per_type", {}).items():
            auc_str = f", AUC: {m_metrics['auc']:.4f}" if "auc" in m_metrics else ""
            print(f"  {m_type}: Acc: {m_metrics['accuracy']:.4f}{auc_str}")

        scheduler.step()

        # Save checkpoint every epoch
        ckpt_path = os.path.join(args.save_dir, f"epoch_{epoch}.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "auc": val_metrics["auc"],
            "accuracy": val_metrics["accuracy"],
        }, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            best_path = os.path.join(args.save_dir, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "auc": best_auc,
                "accuracy": val_metrics["accuracy"],
            }, best_path)
            print(f"New best model! AUC: {best_auc:.4f}")

    # Always save final model
    final_path = os.path.join(args.save_dir, "final_model.pth")
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, final_path)
    print(f"Saved final model: {final_path}")

    print(f"\n--- Final Test ---")
    test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"Test - Acc: {test_metrics['accuracy']:.4f}, AUC: {test_metrics['auc']:.4f}")
    for m_type, m_metrics in test_metrics.get("per_type", {}).items():
        auc_str = f", AUC: {m_metrics['auc']:.4f}" if "auc" in m_metrics else ""
        print(f"  {m_type}: Acc: {m_metrics['accuracy']:.4f}{auc_str}")


if __name__ == "__main__":
    main()
