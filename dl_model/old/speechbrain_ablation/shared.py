import csv
import math
import random
import time
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

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


def build_samples_from_old_all(csv_path: Path, train_audio_dir: Path, test_audio_dir: Path, target_sr=16000):
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


def load_soft_labels(cache_path: Path, train_samples, test_samples):
    payload = torch.load(cache_path, map_location="cpu")
    train_payload = payload["train"]
    test_payload = payload["test"]

    if len(train_payload["teacher_probs"]) != len(train_samples):
        raise ValueError("Cached train soft labels do not match current training sample count.")
    if len(test_payload["teacher_probs"]) != len(test_samples):
        raise ValueError("Cached test soft labels do not match current test sample count.")

    for sample, teacher_prob in zip(train_samples, train_payload["teacher_probs"]):
        sample["teacher_prob"] = float(teacher_prob)
    for sample, teacher_prob in zip(test_samples, test_payload["teacher_probs"]):
        sample["teacher_prob"] = float(teacher_prob)


def soft_distill_loss(student_logits, teacher_prob, temperature):
    teacher_prob = teacher_prob.clamp(1e-5, 1.0 - 1e-5)
    teacher_targets = torch.stack([1.0 - teacher_prob, teacher_prob], dim=1)
    student_log_probs = torch.log_softmax(student_logits / temperature, dim=1)
    loss = -(teacher_targets * student_log_probs).sum(dim=1).mean()
    return loss * (temperature ** 2)


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


def augment_waveforms(wav, noise_std=0.003, gain_low=0.9, gain_high=1.1):
    gain = torch.empty(wav.size(0), 1, device=wav.device).uniform_(gain_low, gain_high)
    wav = wav * gain
    noise = torch.randn_like(wav) * noise_std
    return (wav + noise).clamp_(-1.0, 1.0)


class LogMelFeatureExtractor(nn.Module):
    def __init__(self, sample_rate=16000, n_mels=40, n_fft=400, win_length=400, hop_length=160, f_min=0.0, f_max=None):
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
        return torch.log10(mel.clamp_min(1e-5))


class PairModelBase(nn.Module):
    def __init__(self, sample_rate=16000, n_mels=40, time_mask_max=12, freq_mask_max=6, use_specaugment=True):
        super().__init__()
        self.use_specaugment = use_specaugment
        self.time_mask_max = time_mask_max
        self.freq_mask_max = freq_mask_max
        self.features = LogMelFeatureExtractor(sample_rate=sample_rate, n_mels=n_mels)

    def apply_specaugment(self, feats):
        if not self.training or not self.use_specaugment:
            return feats
        batch, frames, bins = feats.shape
        if self.time_mask_max > 0 and frames > 8:
            for i in range(batch):
                mask = int(torch.randint(0, self.time_mask_max + 1, (1,), device=feats.device).item())
                if 0 < mask < frames:
                    start = int(torch.randint(0, frames - mask + 1, (1,), device=feats.device).item())
                    feats[i, start : start + mask, :] = 0.0
        if self.freq_mask_max > 0 and bins > 4:
            for i in range(batch):
                mask = int(torch.randint(0, self.freq_mask_max + 1, (1,), device=feats.device).item())
                if 0 < mask < bins:
                    start = int(torch.randint(0, bins - mask + 1, (1,), device=feats.device).item())
                    feats[i, :, start : start + mask] = 0.0
        return feats

    def normalize_feats(self, wav):
        feats = self.features(wav).transpose(1, 2)
        feats = feats - feats.mean(dim=1, keepdim=True)
        feats = feats / (feats.std(dim=1, keepdim=True) + 1e-5)
        return self.apply_specaugment(feats)


class TDNNPairStudent(PairModelBase):
    def __init__(
        self,
        sample_rate=16000,
        n_mels=40,
        channels=(128, 192, 256, 256),
        emb_dim=192,
        dropout=0.15,
        time_mask_max=12,
        freq_mask_max=6,
        use_dilation=True,
        use_stats_pooling=True,
        use_pairwise_product=True,
        use_specaugment=True,
    ):
        super().__init__(sample_rate, n_mels, time_mask_max, freq_mask_max, use_specaugment)
        self.use_stats_pooling = use_stats_pooling
        self.use_pairwise_product = use_pairwise_product
        tdnn_layers = []
        in_channels = n_mels
        kernel_sizes = (5, 3, 3, 1)
        dilations = (1, 2, 3, 1) if use_dilation else (1, 1, 1, 1)
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
        pooled_dim = channels[-1] * (2 if use_stats_pooling else 1)
        self.proj = nn.Sequential(
            nn.Linear(pooled_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.embedding_norm = nn.LayerNorm(emb_dim)
        pair_dim = emb_dim * (3 if use_pairwise_product else 2)
        self.classifier = nn.Sequential(
            nn.Linear(pair_dim, emb_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim * 2, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, 2),
        )

    def pool(self, x):
        mean = x.mean(dim=2)
        if not self.use_stats_pooling:
            return mean
        std = torch.sqrt(x.var(dim=2, unbiased=False) + 1e-5)
        return torch.cat([mean, std], dim=1)

    def encode(self, wav):
        feats = self.normalize_feats(wav)
        x = self.tdnn(feats.transpose(1, 2))
        x = self.pool(x)
        x = self.proj(x)
        return self.embedding_norm(x)

    def forward(self, left_audio, right_audio):
        left_emb = self.encode(left_audio)
        right_emb = self.encode(right_audio)
        pair_parts = [0.5 * (left_emb + right_emb), torch.abs(left_emb - right_emb)]
        if self.use_pairwise_product:
            pair_parts.append(left_emb * right_emb)
        return self.classifier(torch.cat(pair_parts, dim=1))


class EmbeddingClassifierStudent(TDNNPairStudent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        emb_dim = kwargs.get("emb_dim", 192)
        dropout = kwargs.get("dropout", 0.15)
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim * 2, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, 2),
        )

    def forward(self, left_audio, right_audio):
        left_emb = self.encode(left_audio)
        right_emb = self.encode(right_audio)
        return self.classifier(torch.cat([left_emb, right_emb], dim=1))


class MLPPairStudent(PairModelBase):
    def __init__(self, sample_rate=16000, n_mels=40, emb_dim=192, dropout=0.15, **kwargs):
        super().__init__(sample_rate, n_mels, kwargs.get("time_mask_max", 12), kwargs.get("freq_mask_max", 6), kwargs.get("use_specaugment", True))
        self.side_mlp = nn.Sequential(
            nn.Linear(n_mels * 2, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim * 3, emb_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim * 2, 2),
        )

    def encode(self, wav):
        feats = self.normalize_feats(wav)
        mean = feats.mean(dim=1)
        std = feats.std(dim=1)
        return self.side_mlp(torch.cat([mean, std], dim=1))

    def forward(self, left_audio, right_audio):
        left_emb = self.encode(left_audio)
        right_emb = self.encode(right_audio)
        return self.classifier(torch.cat([0.5 * (left_emb + right_emb), torch.abs(left_emb - right_emb), left_emb * right_emb], dim=1))


class CNNPairStudent(PairModelBase):
    def __init__(self, sample_rate=16000, n_mels=40, emb_dim=192, dropout=0.15, **kwargs):
        super().__init__(sample_rate, n_mels, kwargs.get("time_mask_max", 12), kwargs.get("freq_mask_max", 6), kwargs.get("use_specaugment", True))
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(128, emb_dim), nn.ReLU(), nn.Dropout(dropout))
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim * 3, emb_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim * 2, 2),
        )

    def encode(self, wav):
        feats = self.normalize_feats(wav).transpose(1, 2).unsqueeze(1)
        return self.proj(self.encoder(feats))

    def forward(self, left_audio, right_audio):
        left_emb = self.encode(left_audio)
        right_emb = self.encode(right_audio)
        return self.classifier(torch.cat([0.5 * (left_emb + right_emb), torch.abs(left_emb - right_emb), left_emb * right_emb], dim=1))


class TransformerPairStudent(PairModelBase):
    def __init__(self, sample_rate=16000, n_mels=40, emb_dim=192, dropout=0.15, **kwargs):
        super().__init__(sample_rate, n_mels, kwargs.get("time_mask_max", 12), kwargs.get("freq_mask_max", 6), kwargs.get("use_specaugment", True))
        self.in_proj = nn.Linear(n_mels, emb_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=4,
            dim_feedforward=emb_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim * 3, emb_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim * 2, 2),
        )

    def encode(self, wav):
        feats = self.normalize_feats(wav)
        x = self.encoder(self.in_proj(feats))
        mean = x.mean(dim=1)
        std = x.std(dim=1)
        return 0.5 * (mean + std)

    def forward(self, left_audio, right_audio):
        left_emb = self.encode(left_audio)
        right_emb = self.encode(right_audio)
        return self.classifier(torch.cat([0.5 * (left_emb + right_emb), torch.abs(left_emb - right_emb), left_emb * right_emb], dim=1))


class ResidualBlock1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.block(x) + x)


class ResNetPairStudent(PairModelBase):
    def __init__(self, sample_rate=16000, n_mels=40, emb_dim=192, dropout=0.15, **kwargs):
        super().__init__(sample_rate, n_mels, kwargs.get("time_mask_max", 12), kwargs.get("freq_mask_max", 6), kwargs.get("use_specaugment", True))
        self.stem = nn.Sequential(
            nn.Conv1d(n_mels, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.res_blocks = nn.Sequential(ResidualBlock1D(128), ResidualBlock1D(128), ResidualBlock1D(128))
        self.proj = nn.Sequential(
            nn.Linear(256, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim * 3, emb_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim * 2, 2),
        )

    def encode(self, wav):
        feats = self.normalize_feats(wav).transpose(1, 2)
        x = self.res_blocks(self.stem(feats))
        pooled = torch.cat([x.mean(dim=2), torch.sqrt(x.var(dim=2, unbiased=False) + 1e-5)], dim=1)
        return self.proj(pooled)

    def forward(self, left_audio, right_audio):
        left_emb = self.encode(left_audio)
        right_emb = self.encode(right_audio)
        return self.classifier(torch.cat([0.5 * (left_emb + right_emb), torch.abs(left_emb - right_emb), left_emb * right_emb], dim=1))


class SincConv1d(nn.Module):
    def __init__(self, out_channels=64, kernel_size=251, sample_rate=16000, min_low_hz=50.0, min_band_hz=50.0):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("SincConv1d kernel_size must be odd.")
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        low_hz = 30.0
        high_hz = sample_rate / 2.0 - (min_low_hz + min_band_hz)
        hz = np.linspace(low_hz, high_hz, out_channels + 1, dtype=np.float32)
        self.low_hz_ = nn.Parameter(torch.from_numpy(hz[:-1]).view(-1, 1))
        self.band_hz_ = nn.Parameter(torch.from_numpy(np.diff(hz)).view(-1, 1))

        n_lin = torch.arange(-(kernel_size // 2), 0, dtype=torch.float32)
        self.register_buffer("n_lin", n_lin / sample_rate, persistent=False)
        self.register_buffer("window", torch.hamming_window(kernel_size, periodic=False, dtype=torch.float32), persistent=False)

    def forward(self, wav):
        device = wav.device
        low = self.min_low_hz + torch.abs(self.low_hz_.to(device))
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_.to(device)), self.min_low_hz, self.sample_rate / 2.0 - 1.0)
        band = (high - low).clamp_min(1.0)

        n = 2 * math.pi * self.n_lin.to(device)
        low_pass1 = 2 * high * torch.sinc(high * n / math.pi)
        low_pass2 = 2 * low * torch.sinc(low * n / math.pi)
        band_pass_left = (low_pass1 - low_pass2) / (2 * band)
        band_pass_center = torch.ones((self.out_channels, 1), device=device)
        band_pass = torch.cat([band_pass_left, band_pass_center, torch.flip(band_pass_left, dims=[1])], dim=1)
        band_pass = band_pass * self.window.to(device).unsqueeze(0)
        band_pass = band_pass / (band_pass.abs().sum(dim=1, keepdim=True) + 1e-6)
        return torch.conv1d(wav.unsqueeze(1), band_pass.unsqueeze(1), stride=1, padding=self.kernel_size // 2)


class SincNetPairStudent(nn.Module):
    def __init__(self, sample_rate=16000, emb_dim=192, dropout=0.15, sinc_channels=64, **kwargs):
        super().__init__()
        self.sample_rate = sample_rate
        self.sinc = SincConv1d(out_channels=sinc_channels, kernel_size=251, sample_rate=sample_rate)
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(sinc_channels),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(3),
            nn.Conv1d(sinc_channels, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(3),
            nn.Conv1d(128, 192, kernel_size=3, padding=1),
            nn.BatchNorm1d(192),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(emb_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim * 3, emb_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim * 2, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, 2),
        )

    def normalize_wav(self, wav):
        wav = wav - wav.mean(dim=1, keepdim=True)
        return wav / (wav.std(dim=1, keepdim=True) + 1e-5)

    def encode(self, wav):
        wav = self.normalize_wav(wav)
        x = self.sinc(wav)
        x = torch.abs(x)
        x = torch.log1p(x)
        x = self.encoder(x)
        return self.proj(x)

    def forward(self, left_audio, right_audio):
        left_emb = self.encode(left_audio)
        right_emb = self.encode(right_audio)
        return self.classifier(torch.cat([0.5 * (left_emb + right_emb), torch.abs(left_emb - right_emb), left_emb * right_emb], dim=1))


class SincTDNNPairStudent(TDNNPairStudent):
    def __init__(
        self,
        sample_rate=16000,
        channels=(128, 192, 256, 256),
        emb_dim=192,
        dropout=0.15,
        time_mask_max=12,
        freq_mask_max=6,
        use_dilation=True,
        use_stats_pooling=True,
        use_pairwise_product=True,
        use_specaugment=True,
        sinc_channels=80,
    ):
        super().__init__(
            sample_rate=sample_rate,
            n_mels=sinc_channels,
            channels=channels,
            emb_dim=emb_dim,
            dropout=dropout,
            time_mask_max=time_mask_max,
            freq_mask_max=freq_mask_max,
            use_dilation=use_dilation,
            use_stats_pooling=use_stats_pooling,
            use_pairwise_product=use_pairwise_product,
            use_specaugment=use_specaugment,
        )
        self.sinc = SincConv1d(out_channels=sinc_channels, kernel_size=251, sample_rate=sample_rate)
        self.sinc_norm = nn.BatchNorm1d(sinc_channels)

    def normalize_wav(self, wav):
        wav = wav - wav.mean(dim=1, keepdim=True)
        return wav / (wav.std(dim=1, keepdim=True) + 1e-5)

    def encode(self, wav):
        wav = self.normalize_wav(wav)
        x = self.sinc(wav)
        x = torch.log1p(torch.abs(x))
        x = self.sinc_norm(x)
        feats = x.transpose(1, 2)
        feats = feats - feats.mean(dim=1, keepdim=True)
        feats = feats / (feats.std(dim=1, keepdim=True) + 1e-5)
        feats = self.apply_specaugment(feats)
        x = self.tdnn(feats.transpose(1, 2))
        x = self.pool(x)
        x = self.proj(x)
        return self.embedding_norm(x)
