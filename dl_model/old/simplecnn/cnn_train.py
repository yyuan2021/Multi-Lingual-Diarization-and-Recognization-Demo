import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =========================
# Dataset
# =========================
class FeatureCacheDataset:
    def __init__(self, path):
        self.records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    self.records.append(json.loads(line))
                except:
                    continue


class PairDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        l = torch.tensor(r["feature"]["left"]["embedding"], dtype=torch.float32)
        r_ = torch.tensor(r["feature"]["right"]["embedding"], dtype=torch.float32)
        x = torch.stack([l, r_], dim=0)
        y = torch.tensor(int(r["is_switch"]), dtype=torch.long)
        return x, y


# =========================
# Model
# =========================
class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 64, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, 3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class SwitchCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = CNNEncoder()
        self.fc = nn.Sequential(
            nn.Linear(256 * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        l = self.enc(x[:, 0].unsqueeze(1))
        r = self.enc(x[:, 1].unsqueeze(1))
        d = torch.abs(l - r)
        return self.fc(torch.cat([l, r, d], dim=1))


# =========================
# Utils
# =========================
def split_train_test(records, seed=42):
    random.Random(seed).shuffle(records)
    n = len(records)
    return records[:int(n * 0.8)], records[int(n * 0.8):]


def balance(records):
    same = [r for r in records if not r["is_switch"]]
    switch = [r for r in records if r["is_switch"]]
    n = min(len(same), len(switch))
    balanced = same[:n] + switch[:n]
    random.shuffle(balanced)
    return balanced


def evaluate(model, loader, device, loss_fn):
    model.eval()
    total_loss = 0
    total = 0
    correct = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = loss_fn(logits, y)

            total_loss += loss.item() * x.size(0)
            total += x.size(0)

            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()

    return total_loss / total, correct / total


# =========================
# Train
# =========================
def train():
    base = Path(__file__).parent
    data_path = base / "mlp_feature_cache.jsonl"
    model_path = base / "best.pth"

    batch_size = 64
    lr = 3e-4
    epochs = 50
    early_stop = 8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    data = FeatureCacheDataset(data_path).records

    train_r, test_r = split_train_test(data)

    # 🔴 1:1
    train_r = balance(train_r)
    test_r = balance(test_r)

    print("train size:", len(train_r))
    print("test size :", len(test_r))

    train_loader = DataLoader(PairDataset(train_r), batch_size, shuffle=True)
    test_loader = DataLoader(PairDataset(test_r), batch_size)

    model = SwitchCNN().to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", patience=3, factor=0.5, min_lr=1e-6
    )

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_loss = float("inf")
    no_improve = 0

    for e in range(epochs):
        model.train()
        train_loss_sum = 0
        train_total = 0
        train_correct = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = loss_fn(logits, y)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            train_loss_sum += loss.item() * x.size(0)
            train_total += x.size(0)

            preds = logits.argmax(dim=1)
            train_correct += (preds == y).sum().item()

        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total

        test_loss, test_acc = evaluate(model, test_loader, device, loss_fn)

        scheduler.step(train_loss)

        print(
            f"{e+1:03d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} | "
            f"lr={opt.param_groups[0]['lr']:.6f}"
        )

        # 🔴 保存 best
        if train_loss < best_loss:
            best_loss = train_loss
            no_improve = 0
            torch.save(model.state_dict(), model_path)
        else:
            no_improve += 1

        if no_improve >= early_stop:
            print("early stop")
            break

    print(f"\nBest model saved at: {model_path}")


if __name__ == "__main__":
    train()