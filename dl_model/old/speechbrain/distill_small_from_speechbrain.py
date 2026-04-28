import argparse
import csv
import random
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import librosa

from baseline.common import compute_metrics, load_audio


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_bool_label(value):
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes"}:
        return 1
    if text in {"0", "false", "f", "no"}:
        return 0
    raise ValueError(f"Unsupported is_switch value: {value}")


def split_pair_from_full_clip(wav):
    mid = len(wav) // 2
    left = wav[:mid].astype(np.float32)
    right = wav[mid:].astype(np.float32)
    return left, right


def build_samples_from_old_all(
    csv_path: Path,
    train_audio_dir: Path,
    test_audio_dir: Path,
    target_sr=16000,
):
    samples = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            split = row["split"].strip().lower()
            audio_dir = train_audio_dir if split == "train" else test_audio_dir
            audio_file = audio_dir / f"{idx + 1}.wav"
            if not audio_file.exists():
                continue
            wav, _ = load_audio(audio_file, sr=target_sr)
            left, right = split_pair_from_full_clip(wav)
            samples.append(
                {
                    "audio_path": row["audio_path"],
                    "source_index": idx + 1,
                    "source_split": split,
                    "left_audio": left,
                    "right_audio": right,
                    "label": parse_bool_label(row["is_switch"]),
                }
            )
    return samples


def build_samples_from_new_extracted(csv_path: Path, test_audio_dir: Path, target_sr=16000, split="test"):
    samples = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            row_split = row.get("split", "").strip().lower()
            if split and row_split and row_split != split:
                continue
            file_index = int(row["test_row_index"]) if row.get("test_row_index") else idx + 1
            audio_file = test_audio_dir / f"{file_index}.wav"
            if not audio_file.exists():
                continue
            wav, _ = load_audio(audio_file, sr=target_sr)
            left, right = split_pair_from_full_clip(wav)
            samples.append(
                {
                    "audio_path": row["audio_path"],
                    "test_row_index": file_index,
                    "left_audio": left,
                    "right_audio": right,
                    "label": parse_bool_label(row["is_switch"]),
                }
            )
    return samples


def collate_audio_pairs(batch):
    left_lengths = [len(item["left_audio"]) for item in batch]
    right_lengths = [len(item["right_audio"]) for item in batch]
    max_left = max(left_lengths)
    max_right = max(right_lengths)

    left_batch = torch.zeros(len(batch), max_left, dtype=torch.float32)
    right_batch = torch.zeros(len(batch), max_right, dtype=torch.float32)
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    teacher_probs = torch.tensor([item["teacher_prob"] for item in batch], dtype=torch.float32)

    for i, item in enumerate(batch):
        left = torch.from_numpy(item["left_audio"])
        right = torch.from_numpy(item["right_audio"])
        left_batch[i, : left.numel()] = left
        right_batch[i, : right.numel()] = right

    return {
        "left_audio": left_batch,
        "right_audio": right_batch,
        "labels": labels,
        "teacher_prob": teacher_probs,
    }


class DistillationPairDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class LogMelFeatureExtractor(nn.Module):
    def __init__(
        self,
        sample_rate=16000,
        n_mels=40,
        n_fft=400,
        win_length=400,
        hop_length=160,
        f_min=0.0,
        f_max=None,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)
        mel_filter = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=f_min,
            fmax=f_max,
        ).astype(np.float32)
        self.register_buffer("mel_filter", torch.from_numpy(mel_filter), persistent=False)

    def forward(self, wav):
        spec = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )
        power = spec.abs().pow(2.0)
        mel = torch.matmul(self.mel_filter.unsqueeze(0), power)
        log_mel = torch.log10(mel.clamp_min(1e-5))
        return log_mel


class SmallSpeakerStudent(nn.Module):
    def __init__(
        self,
        sample_rate=16000,
        n_mels=40,
        channels=(128, 192, 256, 256),
        emb_dim=192,
        dropout=0.15,
        time_mask_max=12,
        freq_mask_max=6,
    ):
        super().__init__()
        self.time_mask_max = time_mask_max
        self.freq_mask_max = freq_mask_max
        self.features = LogMelFeatureExtractor(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=400,
            win_length=400,
            hop_length=160,
        )
        tdnn_layers = []
        in_channels = n_mels
        kernel_sizes = (5, 3, 3, 1)
        dilations = (1, 2, 3, 1)
        for out_channels, kernel_size, dilation in zip(channels, kernel_sizes, dilations):
            tdnn_layers.extend(
                [
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        padding=((kernel_size - 1) // 2) * dilation,
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                ]
            )
            in_channels = out_channels
        self.tdnn = nn.Sequential(*tdnn_layers)
        pooled_dim = channels[-1] * 2
        self.proj = nn.Sequential(
            nn.Linear(pooled_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.embedding_norm = nn.LayerNorm(emb_dim)
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim * 3, emb_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim * 2, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, 2),
        )

    @staticmethod
    def _stats_pool(x):
        mean = x.mean(dim=2)
        std = torch.sqrt(x.var(dim=2, unbiased=False) + 1e-5)
        return torch.cat([mean, std], dim=1)

    def _apply_specaugment(self, feats):
        if not self.training:
            return feats

        batch, frames, bins = feats.shape
        if self.time_mask_max > 0 and frames > 8:
            for i in range(batch):
                mask = int(torch.randint(0, self.time_mask_max + 1, (1,), device=feats.device).item())
                if mask > 0 and mask < frames:
                    start = int(torch.randint(0, frames - mask + 1, (1,), device=feats.device).item())
                    feats[i, start : start + mask, :] = 0.0

        if self.freq_mask_max > 0 and bins > 4:
            for i in range(batch):
                mask = int(torch.randint(0, self.freq_mask_max + 1, (1,), device=feats.device).item())
                if mask > 0 and mask < bins:
                    start = int(torch.randint(0, bins - mask + 1, (1,), device=feats.device).item())
                    feats[i, :, start : start + mask] = 0.0

        return feats

    def encode(self, wav):
        feats = self.features(wav)
        feats = feats.transpose(1, 2)
        feats = feats - feats.mean(dim=1, keepdim=True)
        feats = feats / (feats.std(dim=1, keepdim=True) + 1e-5)
        feats = self._apply_specaugment(feats)
        x = feats.transpose(1, 2)
        x = self.tdnn(x)
        x = self._stats_pool(x)
        x = self.proj(x)
        return self.embedding_norm(x)

    def forward(self, left_audio, right_audio):
        left_emb = self.encode(left_audio)
        right_emb = self.encode(right_audio)
        pair_feat = torch.cat(
            [
                0.5 * (left_emb + right_emb),
                torch.abs(left_emb - right_emb),
                left_emb * right_emb,
            ],
            dim=1,
        )
        return self.classifier(pair_feat)


def augment_waveforms(wav, noise_std=0.003, gain_low=0.9, gain_high=1.1):
    gain = torch.empty(wav.size(0), 1, device=wav.device).uniform_(gain_low, gain_high)
    wav = wav * gain
    noise = torch.randn_like(wav) * noise_std
    return (wav + noise).clamp_(-1.0, 1.0)


def load_soft_labels(cache_path: Path, train_samples, test_samples):
    payload = torch.load(cache_path, map_location="cpu")
    train_payload = payload["train"]
    test_payload = payload["test"]

    if len(train_payload["teacher_probs"]) != len(train_samples):
        raise ValueError("Cached train soft labels do not match current training sample count.")
    if len(test_payload["teacher_probs"]) != len(test_samples):
        raise ValueError("Cached test soft labels do not match current test sample count.")

    train_labels = [int(sample["label"]) for sample in train_samples]
    test_labels = [int(sample["label"]) for sample in test_samples]
    if train_payload["labels"] != train_labels:
        raise ValueError("Cached train soft labels do not match current training labels.")
    if test_payload["labels"] != test_labels:
        raise ValueError("Cached test soft labels do not match current test labels.")

    for sample, teacher_prob in zip(train_samples, train_payload["teacher_probs"]):
        sample["teacher_prob"] = float(teacher_prob)
    for sample, teacher_prob in zip(test_samples, test_payload["teacher_probs"]):
        sample["teacher_prob"] = float(teacher_prob)

    print(f"Loaded cached teacher soft labels from: {cache_path}")
    return {
        "train_teacher_preds": train_payload.get("teacher_preds"),
        "test_teacher_preds": test_payload.get("teacher_preds"),
    }


def soft_distill_loss(student_logits, teacher_prob, temperature):
    teacher_prob = teacher_prob.clamp(1e-5, 1.0 - 1e-5)
    teacher_targets = torch.stack([1.0 - teacher_prob, teacher_prob], dim=1)
    student_log_probs = torch.log_softmax(student_logits / temperature, dim=1)
    loss = -(teacher_targets * student_log_probs).sum(dim=1).mean()
    return loss * (temperature ** 2)


def consistency_loss(logits_a, logits_b):
    probs_a = torch.softmax(logits_a, dim=1)
    probs_b = torch.softmax(logits_b, dim=1)
    return torch.mean((probs_a - probs_b) ** 2)


class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.num_updates = 0
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.state_dict().items()
        }

    def update(self, model):
        with torch.no_grad():
            self.num_updates += 1
            for name, param in model.state_dict().items():
                if self.num_updates == 1:
                    self.shadow[name].copy_(param.detach())
                elif torch.is_floating_point(self.shadow[name]):
                    self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)
                else:
                    self.shadow[name].copy_(param.detach())

    def apply_to(self, model):
        model.load_state_dict(self.shadow, strict=True)


def evaluate_student(model, loader, device, ce_loss_fn, use_tta_swap=False):
    model.eval()
    total_loss = 0.0
    labels_all = []
    preds_all = []

    with torch.no_grad():
        for batch in loader:
            left = batch["left_audio"].to(device)
            right = batch["right_audio"].to(device)
            labels = batch["labels"].to(device)

            logits = model(left, right)
            if use_tta_swap:
                logits = 0.5 * (logits + model(right, left))
            loss = ce_loss_fn(logits, labels)
            total_loss += loss.item() * labels.size(0)

            preds = logits.argmax(dim=1)
            labels_all.extend(labels.cpu().tolist())
            preds_all.extend(preds.cpu().tolist())

    metrics = compute_metrics(labels_all, preds_all)
    metrics["loss"] = total_loss / len(labels_all)
    return metrics


def benchmark_student(model, dataset, device, limit=200):
    model.eval()
    count = min(limit, len(dataset))
    if count == 0:
        return None
    start = time.perf_counter()
    with torch.no_grad():
        for idx in range(count):
            sample = dataset[idx]
            left = torch.from_numpy(sample["left_audio"]).unsqueeze(0).to(device)
            right = torch.from_numpy(sample["right_audio"]).unsqueeze(0).to(device)
            _ = model(left, right)
    elapsed = time.perf_counter() - start
    return elapsed * 1000.0 / count


def train(args):
    set_seed(args.seed)
    root = Path(__file__).resolve().parents[2]

    old_csv = root / args.old_csv
    old_train_audio_dir = root / args.old_train_audio_dir
    old_test_audio_dir = root / args.old_test_audio_dir
    new_test_csv = root / args.new_test_csv
    new_test_audio_dir = root / args.new_test_audio_dir
    checkpoint_dir = root / args.checkpoint_dir
    soft_labels_cache = root / args.soft_labels_cache
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_samples = build_samples_from_old_all(
        old_csv,
        old_train_audio_dir,
        old_test_audio_dir,
        target_sr=args.sr,
    )
    test_samples = build_samples_from_new_extracted(
        new_test_csv,
        new_test_audio_dir,
        target_sr=args.sr,
        split="test",
    )

    train_pos = sum(sample["label"] for sample in train_samples)
    test_pos = sum(sample["label"] for sample in test_samples)
    print(f"Train samples (old train + old test): {len(train_samples)}")
    print(f"  positives={train_pos} negatives={len(train_samples) - train_pos}")
    print(f"Test samples (new testset): {len(test_samples)}")
    print(f"  positives={test_pos} negatives={len(test_samples) - test_pos}")
    if not train_samples or not test_samples:
        raise RuntimeError("Training or test samples are empty after loading windows.")

    if not soft_labels_cache.exists():
        raise FileNotFoundError(
            f"Soft label cache not found: {soft_labels_cache}. "
            "Please provide an existing cache file generated earlier."
        )
    soft_label_meta = load_soft_labels(soft_labels_cache, train_samples, test_samples)

    teacher_metrics = None
    teacher_metric_label = None
    teacher_labels = [sample["label"] for sample in test_samples]
    cached_teacher_preds = soft_label_meta.get("test_teacher_preds")
    if cached_teacher_preds is not None:
        teacher_metrics = compute_metrics(teacher_labels, cached_teacher_preds)
        teacher_metric_label = "Teacher SpeechBrain ECAPA on new testset"
    else:
        proxy_teacher_preds = [1 if sample["teacher_prob"] >= 0.5 else 0 for sample in test_samples]
        teacher_metrics = compute_metrics(teacher_labels, proxy_teacher_preds)
        teacher_metric_label = "Teacher soft-label proxy on new testset"
        print(
            "\nCached soft labels do not include the teacher's original decisions, "
            "so the teacher metrics below are only a 0.5-threshold proxy and may be much lower "
            "than the real teacher accuracy you measured earlier."
        )
    print(
        f"\n{teacher_metric_label} | "
        f"acc={teacher_metrics['accuracy']:.4f} "
        f"precision={teacher_metrics['precision']:.4f} "
        f"recall={teacher_metrics['recall']:.4f} "
        f"f1={teacher_metrics['f1']:.4f}"
    )

    train_dataset = DistillationPairDataset(train_samples)
    test_dataset = DistillationPairDataset(test_samples)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_audio_pairs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_audio_pairs,
    )

    if args.student_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.student_device)
    print(f"student_device: {device}")

    model = SmallSpeakerStudent(
        sample_rate=args.sr,
        n_mels=args.n_mels,
        channels=tuple(args.student_channels),
        emb_dim=args.emb_dim,
        dropout=args.dropout,
        time_mask_max=args.time_mask_max,
        freq_mask_max=args.freq_mask_max,
    ).to(device)
    use_amp = bool(args.amp and device.type == "cuda")
    param_count = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"student_params: total={param_count:,} trainable={trainable_params:,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr,
    )
    ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.03)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ema = ModelEMA(model, decay=args.ema_decay) if args.use_ema else None
    eval_model = model

    best_f1 = -1.0
    best_acc = 0.0
    best_epoch = 0
    no_improve = 0
    best_path = checkpoint_dir / args.best_name
    final_path = checkpoint_dir / args.final_name

    print("\nStarting student distillation...")
    print(f"epochs={args.epochs} batch_size={args.batch_size} lr={args.lr}")
    print(f"alpha={args.alpha} temperature={args.temperature} patience={args.patience}")
    print(f"soft_labels_cache={soft_labels_cache}")
    print(f"amp={'on' if use_amp else 'off'}")
    print(
        f"use_ema={args.use_ema} ema_decay={args.ema_decay} "
        f"consistency_lambda={args.consistency_lambda}"
    )
    print(
        "student_architecture="
        f"mel_tdnn channels={tuple(args.student_channels)} emb_dim={args.emb_dim}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        train_labels = []
        train_preds = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            left = batch["left_audio"].to(device)
            right = batch["right_audio"].to(device)
            labels = batch["labels"].to(device)
            teacher_prob = batch["teacher_prob"].to(device)
            if args.waveform_aug:
                left = augment_waveforms(left)
                right = augment_waveforms(right)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(left, right)
                logits_swapped = model(right, left)
                hard_loss = 0.5 * (
                    ce_loss_fn(logits, labels) + ce_loss_fn(logits_swapped, labels)
                )
                distill_loss = 0.5 * (
                    soft_distill_loss(logits, teacher_prob, args.temperature)
                    + soft_distill_loss(logits_swapped, teacher_prob, args.temperature)
                )
                sym_loss = consistency_loss(logits, logits_swapped)
                loss = (
                    args.alpha * hard_loss
                    + (1.0 - args.alpha) * distill_loss
                    + args.consistency_lambda * sym_loss
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)

            train_loss_sum += loss.item() * labels.size(0)
            train_count += labels.size(0)
            preds = logits.argmax(dim=1)
            train_labels.extend(labels.cpu().tolist())
            train_preds.extend(preds.cpu().tolist())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_metrics = compute_metrics(train_labels, train_preds)
        train_loss = train_loss_sum / train_count
        if ema is not None:
            ema.apply_to(eval_model)
        test_metrics = evaluate_student(
            eval_model,
            test_loader,
            device,
            ce_loss_fn,
            use_tta_swap=args.eval_tta_swap,
        )
        scheduler.step()

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} train_acc={train_metrics['accuracy']:.4f} train_f1={train_metrics['f1']:.4f} | "
            f"test_loss={test_metrics['loss']:.4f} test_acc={test_metrics['accuracy']:.4f} "
            f"precision={test_metrics['precision']:.4f} recall={test_metrics['recall']:.4f} f1={test_metrics['f1']:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.6f}"
        )

        current_f1 = test_metrics["f1"] if test_metrics["f1"] is not None else -1.0
        improved = current_f1 > best_f1 or (
            abs(current_f1 - best_f1) < 1e-8 and test_metrics["accuracy"] > best_acc
        )
        if improved:
            best_f1 = current_f1
            best_acc = test_metrics["accuracy"]
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": eval_model.state_dict(),
                    "test_metrics": test_metrics,
                    "teacher_metrics": teacher_metrics,
                    "args": vars(args),
                },
                best_path,
            )
            print(f"Saved best checkpoint to: {best_path}")
        else:
            no_improve += 1
            print(
                f"No improvement for {no_improve} epoch(s). "
                f"Best so far: epoch {best_epoch}, acc={best_acc:.4f}, f1={best_f1:.4f}"
            )
            if no_improve >= args.patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": eval_model.state_dict(),
            "best_epoch": best_epoch,
            "best_acc": best_acc,
            "best_f1": best_f1,
            "teacher_metrics": teacher_metrics,
            "args": vars(args),
        },
        final_path,
    )

    if ema is not None:
        ema.apply_to(eval_model)
    student_ms = benchmark_student(eval_model, test_dataset, device=device, limit=args.benchmark_samples)

    print("\nFinished distillation.")
    print(f"{teacher_metric_label} acc/f1: {teacher_metrics['accuracy']:.4f}/{teacher_metrics['f1']:.4f}")
    print(f"Best student acc/f1: {best_acc:.4f}/{best_f1:.4f} at epoch {best_epoch}")
    if student_ms is not None:
        print(f"Student avg inference ms/sample: {student_ms:.4f}")
    print(f"Best checkpoint : {best_path}")
    print(f"Final checkpoint: {final_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train a small speaker-pair student from cached soft labels using all old samples "
            "for training and the new test set for evaluation."
        )
    )
    parser.add_argument("--old-csv", default="dl_model/baseline_train_test_segments.csv")
    parser.add_argument("--old-train-audio-dir", default="datasets/mlp_train/train")
    parser.add_argument("--old-test-audio-dir", default="datasets/mlp_train/test")
    parser.add_argument(
        "--new-test-csv",
        default="dl_model/baseline_train_test_segments_switchlingua_seame.csv",
    )
    parser.add_argument(
        "--new-test-audio-dir",
        default="datasets/baseline_switchlingua_seame_testset/test",
    )
    parser.add_argument("--checkpoint-dir", default="dl_model/checkpoints")
    parser.add_argument(
        "--best-name",
        default="speechbrain_small_student_distilled_best.pth",
    )
    parser.add_argument(
        "--final-name",
        default="speechbrain_small_student_distilled_final.pth",
    )
    parser.add_argument("--student-device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--consistency-lambda", type=float, default=0.0)
    parser.add_argument("--use-ema", action="store_true", default=False)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--n-mels", type=int, default=40)
    parser.add_argument("--emb-dim", type=int, default=192)
    parser.add_argument("--student-channels", type=int, nargs="+", default=[128, 192, 256, 256])
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--time-mask-max", type=int, default=12)
    parser.add_argument("--freq-mask-max", type=int, default=6)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--waveform-aug", action="store_true", default=True)
    parser.add_argument("--no-waveform-aug", dest="waveform_aug", action="store_false")
    parser.add_argument("--eval-tta-swap", action="store_true", default=True)
    parser.add_argument("--no-eval-tta-swap", dest="eval_tta_swap", action="store_false")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--benchmark-samples", type=int, default=100)
    parser.add_argument(
        "--soft-labels-cache",
        default="dl_model/checkpoints/speechbrain_soft_labels_old_all_eval_new.pt",
    )
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
