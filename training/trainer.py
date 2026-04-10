"""Training and evaluation logic for temporal deepfake detector."""
import torch
from torch.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm


def train_epoch(model, loader, criterion, optimizer, device, accum_steps=1):
    model.train()
    running_losses = {}
    all_preds = []
    all_labels = []

    use_amp = device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)

    optimizer.zero_grad()
    for step, (view1, view2, labels, _) in enumerate(tqdm(loader, desc="Training")):
        view1 = view1.to(device)   # [B, T, 3, H, W]
        view2 = view2.to(device)   # [B, T, 3, H, W]
        labels = labels.to(device)

        with autocast("cuda", enabled=use_amp):
            video_logits, frame_logits, feat_v1, feat_v2 = model(view1, view2)
            loss, loss_dict = criterion(video_logits, frame_logits, labels, feat_v1, feat_v2)
            loss = loss / accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        for k, v in loss_dict.items():
            running_losses[k] = running_losses.get(k, 0) + v

        preds = video_logits.argmax(dim=1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().tolist())

    n_batches = len(loader)
    avg_losses = {k: v / n_batches for k, v in running_losses.items()}
    acc = accuracy_score(all_labels, all_preds)

    return avg_losses, acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_losses = {}
    all_preds = []
    all_probs = []
    all_labels = []
    all_mtypes = []

    use_amp = device.type == "cuda"

    for view1, view2, labels, m_types in tqdm(loader, desc="Evaluating"):
        view1 = view1.to(device)
        view2 = view2.to(device)
        labels = labels.to(device)

        with autocast("cuda", enabled=use_amp):
            video_logits, frame_logits, feat_v1, feat_v2 = model(view1, view2)
            loss, loss_dict = criterion(video_logits, frame_logits, labels, feat_v1, feat_v2)

        for k, v in loss_dict.items():
            running_losses[k] = running_losses.get(k, 0) + v

        probs = torch.softmax(video_logits, dim=1)[:, 1].cpu()
        preds = video_logits.argmax(dim=1).cpu()
        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().tolist())
        all_mtypes.extend(m_types)

    n_batches = len(loader)
    avg_losses = {k: v / n_batches for k, v in running_losses.items()}

    metrics = {"loss": avg_losses}
    metrics["accuracy"] = accuracy_score(all_labels, all_preds)

    if len(set(all_labels)) > 1:
        metrics["auc"] = roc_auc_score(all_labels, all_probs)
    else:
        metrics["auc"] = 0.0

    # Per manipulation type metrics
    per_type = {}
    for m_type in set(all_mtypes):
        idxs = [i for i, m in enumerate(all_mtypes) if m == m_type]
        if len(idxs) < 2:
            continue
        m_labels = [all_labels[i] for i in idxs]
        m_preds = [all_preds[i] for i in idxs]
        m_probs = [all_probs[i] for i in idxs]
        per_type[m_type] = {
            "accuracy": accuracy_score(m_labels, m_preds),
        }
        if len(set(m_labels)) > 1:
            per_type[m_type]["auc"] = roc_auc_score(m_labels, m_probs)
    metrics["per_type"] = per_type

    return metrics
