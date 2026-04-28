import csv
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

class FeatureCacheDataset(Dataset):
    def __init__(self, cache_path):
        self.cache_path = Path(cache_path)
        self.cache_records = []
        self.skipped_lines = []

        with self.cache_path.open("r", encoding="utf-8") as cache_file:
            for line_no, raw_line in enumerate(cache_file, start=1):
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    self.cache_records.append(json.loads(raw_line))
                except json.JSONDecodeError as exc:
                    self.skipped_lines.append((line_no, str(exc)))

        if not self.cache_records:
            raise ValueError(f"No valid records found in {self.cache_path}")

    def __len__(self):
        return len(self.cache_records)

    def __getitem__(self, index):
        return self.cache_records[index]

class PairEmbeddingDataset(Dataset):
    def __init__(self, records):
        self.cache_records = records

    def __len__(self):
        return len(self.cache_records)

    def __getitem__(self, index):
        record = self.cache_records[index]
        left_embedding = torch.tensor(
            record["feature"]["left"]["embedding"], dtype=torch.float32
        )
        right_embedding = torch.tensor(
            record["feature"]["right"]["embedding"], dtype=torch.float32
        )
        pair_tensor = torch.stack([left_embedding, right_embedding], dim=0)  # [2,512]
        label = torch.tensor(int(record["is_switch"]), dtype=torch.long)
        return pair_tensor, label

class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)
        return out

class ResNet18Encoder1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = [BasicBlock1D(in_channels, out_channels, stride=stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock1D(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).squeeze(-1)
        return x

class SwitchResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ResNet18Encoder1D()
        self.classifier = nn.Sequential(
            nn.Linear(512 * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        # x: [B, 2, 512]
        left = x[:, 0].unsqueeze(1)   # [B,1,512]
        right = x[:, 1].unsqueeze(1)

        left_feat = self.encoder(left)
        right_feat = self.encoder(right)
        abs_diff = torch.abs(left_feat - right_feat)

        fused = torch.cat([left_feat, right_feat, abs_diff], dim=1)
        return self.classifier(fused)

def count_labels(records):
    same_count = 0
    switch_count = 0
    for record in records:
        if bool(record["is_switch"]):
            switch_count += 1
        else:
            same_count += 1
    return same_count, switch_count

def get_dataset_name(record):
    audio_path = str(record.get("audio_path") or "").replace("\\", "/").lower()
    if audio_path.startswith("datasets/"):
        parts = audio_path.split("/")
        if len(parts) >= 2 and parts[1]:
            return parts[1]
    return "unknown"

def count_by_dataset(records):
    counter = Counter()
    for record in records:
        counter[get_dataset_name(record)] += 1
    return counter

def balance_records_by_dataset(records, seed):
    dataset_to_records = defaultdict(list)
    for record in records:
        dataset_to_records[get_dataset_name(record)].append(record)

    if not dataset_to_records:
        raise ValueError("No dataset groups found for balancing.")

    dataset_counts = {name: len(rows) for name, rows in dataset_to_records.items()}
    target_count = min(dataset_counts.values())
    if target_count < 1:
        raise ValueError("Each dataset group must contain at least one record.")

    rng = random.Random(seed)
    balanced_records = []
    for dataset_name in sorted(dataset_to_records):
        rows = list(dataset_to_records[dataset_name])
        rng.shuffle(rows)
        balanced_records.extend(rows[:target_count])

    rng.shuffle(balanced_records)
    return balanced_records, dataset_counts, target_count

def select_evenly_spaced_records(records, target_count):
    if target_count >= len(records):
        return list(records)
    if target_count <= 0:
        return []
    if target_count == 1:
        return [records[len(records) // 2]]

    last_index = len(records) - 1
    selected = []
    used_indices = set()

    for step in range(target_count):
        index = round(step * last_index / (target_count - 1))
        while index in used_indices and index < last_index:
            index += 1
        while index in used_indices and index > 0:
            index -= 1
        used_indices.add(index)
        selected.append(records[index])

    return selected

def balance_two_class_records(same_records, switch_records):
    if not same_records or not switch_records:
        raise ValueError("Need both classes to build a 1:1 dataset.")

    minority_count = min(len(same_records), len(switch_records))
    balanced_same = select_evenly_spaced_records(same_records, minority_count)
    balanced_switch = select_evenly_spaced_records(switch_records, minority_count)

    balanced_records = []
    for same_record, switch_record in zip(balanced_same, balanced_switch):
        balanced_records.append(same_record)
        balanced_records.append(switch_record)

    return balanced_records, minority_count

def stratified_split_with_balanced_eval(records, train_ratio, val_ratio, test_ratio, seed):
    same_records = [record for record in records if not bool(record["is_switch"])]
    switch_records = [record for record in records if bool(record["is_switch"])]

    rng = random.Random(seed)
    rng.shuffle(same_records)
    rng.shuffle(switch_records)

    def split_one_class(class_records):
        total_count = len(class_records)
        if total_count < 3:
            raise ValueError("Need at least 3 samples per class for train/val/test split.")

        train_count = int(total_count * train_ratio)
        val_count = int(total_count * val_ratio)

        train_count = max(1, train_count)
        val_count = max(1, val_count)
        test_count = total_count - train_count - val_count

        if test_count < 1:
            test_count = 1
            if train_count >= val_count and train_count > 1:
                train_count -= 1
            elif val_count > 1:
                val_count -= 1

        train_records = class_records[:train_count]
        val_records = class_records[train_count:train_count + val_count]
        test_records = class_records[train_count + val_count:train_count + val_count + test_count]
        return train_records, val_records, test_records

    same_train, same_val, same_test = split_one_class(same_records)
    switch_train, switch_val, switch_test = split_one_class(switch_records)

    val_records, val_balanced_per_class = balance_two_class_records(same_val, switch_val)
    test_records, test_balanced_per_class = balance_two_class_records(same_test, switch_test)

    train_records = same_train + switch_train

    rng.shuffle(train_records)
    rng.shuffle(val_records)
    rng.shuffle(test_records)
    return (
        train_records,
        val_records,
        test_records,
        val_balanced_per_class,
        test_balanced_per_class,
    )

def select_device():
    force_cpu = os.environ.get("FORCE_CPU", "").lower() in {"1", "true", "yes"}
    requested_device = os.environ.get("TRAIN_DEVICE", "").strip().lower()

    if force_cpu:
        device = torch.device("cpu")
    elif requested_device in {"cpu", "cuda"}:
        device = torch.device(requested_device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        try:
            _ = torch.zeros(1).to(device)
        except Exception as exc:
            print("CUDA initialization failed, falling back to CPU.")
            print(f"CUDA error: {exc}")
            device = torch.device("cpu")
    
    return device

def set_global_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for pair_tensor, label in data_loader:
            pair_tensor = pair_tensor.to(device)
            label = label.to(device)

            logits = model(pair_tensor)
            loss = criterion(logits, label)

            total_loss += loss.item() * pair_tensor.size(0)
            predictions = logits.argmax(dim=1)
            total_correct += (predictions == label).sum().item()
            total_count += pair_tensor.size(0)

    average_loss = total_loss / max(total_count, 1)
    accuracy = total_correct / max(total_count, 1)
    return average_loss, accuracy

def predict_records(model, records, device, batch_size):
    dataset = PairEmbeddingDataset(records)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()

    predictions = []
    offset = 0
    with torch.no_grad():
        for pair_tensor, _label in data_loader:
            pair_tensor = pair_tensor.to(device)
            logits = model(pair_tensor)
            probs = torch.softmax(logits, dim=1).cpu()
            pred_ids = torch.argmax(probs, dim=1).tolist()

            batch_size_now = len(pred_ids)
            batch_records = records[offset:offset + batch_size_now]
            for record, pred_id, prob_vec in zip(batch_records, pred_ids, probs.tolist()):
                predictions.append({
                    "audio_path": record.get("audio_path"),
                    "json_path": record.get("json_path"),
                    "dataset": get_dataset_name(record),
                    "true_label_id": int(record.get("is_switch")),
                    "true_label": "code_switch" if bool(record.get("is_switch")) else "speaker_switch",
                    "pred_label_id": int(pred_id),
                    "pred_label": "code_switch" if int(pred_id) == 1 else "speaker_switch",
                    "prob_speaker_switch": float(prob_vec[0]),
                    "prob_code_switch": float(prob_vec[1]),
                })
            offset += batch_size_now

    return predictions

def write_summary_csv(csv_path, summary_row):
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "model",
                "input_type",
                "raw_same_count",
                "raw_switch_count",
                "balanced_per_class",
                "train_records",
                "val_records",
                "test_records",
                "best_epoch",
                "best_val_loss",
                "best_val_acc",
                "test_loss",
                "test_acc",
            ],
        )
        writer.writeheader()
        writer.writerow(summary_row)

def write_split_file_list(output_path, records):
    rows = []
    for record in records:
        rows.append({
            "audio_path": record.get("audio_path"),
            "json_path": record.get("json_path"),
            "dataset": get_dataset_name(record),
            "is_switch": bool(record.get("is_switch")),
        })
    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

def write_prediction_list(output_path, rows):
    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

def train():
    base_dir = Path(__file__).resolve().parent
    cache_path = base_dir / "mlp_feature_cache.jsonl"
    save_path = base_dir / "switch_resnet18_1d.pth"
    summary_csv_path = base_dir / "switch_resnet18_1d_result.csv"
    test_list_path = base_dir / "switch_resnet18_1d_test_files.json"
    test_pred_path = base_dir / "switch_resnet18_1d_test_predictions.json"

    batch_size = 64
    num_epochs = 100
    learning_rate = 3e-4
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    random_seed = 42
    weight_decay = 1e-4
    early_stop_patience = 12

    set_global_seed(random_seed)
    device = select_device()
    print(f"Using device: {device}")
    print(f"Reading data from: {cache_path}")

    raw_dataset = FeatureCacheDataset(cache_path)
    if raw_dataset.skipped_lines:
        print(f"Skipped invalid jsonl lines: {len(raw_dataset.skipped_lines)}")
        for line_no, err in raw_dataset.skipped_lines[:10]:
            print(f"  line {line_no}: {err}")
        if len(raw_dataset.skipped_lines) > 10:
            print("  ...")

    balanced_records, raw_dataset_counts, balanced_per_dataset = balance_records_by_dataset(
        raw_dataset.cache_records,
        seed=random_seed,
    )
    balanced_dataset_counts = count_by_dataset(balanced_records)
    raw_same_count, raw_switch_count = count_labels(balanced_records)
    train_records, val_records, test_records, val_balanced_per_class, test_balanced_per_class = (
        stratified_split_with_balanced_eval(
        balanced_records,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=random_seed,
    )
    )

    train_same_count, train_switch_count = count_labels(train_records)
    val_same_count, val_switch_count = count_labels(val_records)
    test_same_count, test_switch_count = count_labels(test_records)

    train_dataset = PairEmbeddingDataset(train_records)
    val_dataset = PairEmbeddingDataset(val_records)
    test_dataset = PairEmbeddingDataset(test_records)

    write_split_file_list(test_list_path, test_records)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = SwitchResNet18().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    no_improve_epochs = 0

    print(f"Raw dataset counts: {dict(sorted(raw_dataset_counts.items()))}")
    print(
        f"Balanced dataset counts: {dict(sorted(balanced_dataset_counts.items()))} "
        f"(per_dataset={balanced_per_dataset})"
    )
    print(f"Balanced class counts: same={raw_same_count}, switch={raw_switch_count}")
    print(f"Train split: total={len(train_records)} same={train_same_count} switch={train_switch_count}")
    print(
        f"Val split: total={len(val_records)} same={val_same_count} "
        f"switch={val_switch_count} balanced_per_class={val_balanced_per_class}"
    )
    print(
        f"Test split: total={len(test_records)} same={test_same_count} "
        f"switch={test_switch_count} balanced_per_class={test_balanced_per_class}"
    )
    print("Model: ResNet18-style 1D Siamese CNN")
    print("Input shape: [B, 2, 512]")
    print(f"Random seed: {random_seed}")
    print(f"Test file list saved to: {test_list_path}")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_count = 0

        for pair_tensor, label in train_loader:
            pair_tensor = pair_tensor.to(device)
            label = label.to(device)

            logits = model(pair_tensor)
            loss = criterion(logits, label)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * pair_tensor.size(0)
            predictions = logits.argmax(dim=1)
            total_correct += (predictions == label).sum().item()
            total_count += pair_tensor.size(0)

        train_loss = total_loss / max(total_count, 1)
        train_acc = total_correct / max(total_count, 1)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        improved = (val_acc > best_val_acc) or (
            val_acc == best_val_acc and val_loss < best_val_loss
        )
        if improved:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch + 1
            no_improve_epochs = 0
            torch.save(model.state_dict(), save_path)
        else:
            no_improve_epochs += 1

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1:02d}/{num_epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"lr={current_lr:.6f}"
        )

        if no_improve_epochs >= early_stop_patience:
            print(
                f"Early stopping at epoch {epoch + 1}. "
                f"Best epoch was {best_epoch} with val_acc={best_val_acc:.4f}, "
                f"val_loss={best_val_loss:.4f}"
            )
            break

    best_model = SwitchResNet18().to(device)
    best_model.load_state_dict(torch.load(save_path, map_location=device))
    test_loss, test_acc = evaluate(best_model, test_loader, criterion, device)
    test_predictions = predict_records(best_model, test_records, device, batch_size=batch_size)
    write_prediction_list(test_pred_path, test_predictions)

    print(
        f"Best model saved to: {save_path} | "
        f"best_epoch={best_epoch} best_val_acc={best_val_acc:.4f} "
        f"best_val_loss={best_val_loss:.4f}"
    )
    print(f"Final test | test_loss={test_loss:.4f} test_acc={test_acc:.4f}")
    print(f"Test predictions saved to: {test_pred_path}")

    write_summary_csv(
        summary_csv_path,
        {
            "model": "resnet18_1d_siamese",
            "input_type": "pair_embedding_[B,2,512]",
            "raw_same_count": raw_same_count,
            "raw_switch_count": raw_switch_count,
            "balanced_per_class": min(val_balanced_per_class, test_balanced_per_class),
            "train_records": len(train_records),
            "val_records": len(val_records),
            "test_records": len(test_records),
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
        },
    )
    print(f"Summary csv saved to: {summary_csv_path}")

if __name__ == "__main__":
    train()