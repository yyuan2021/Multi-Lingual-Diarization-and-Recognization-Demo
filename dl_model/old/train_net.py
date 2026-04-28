import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from dl_model.old.transformer.model import TransformerModel
from dl_model.old.mlp.model2 import MLPModel2
from dl_model.old.mlp.model import MLPModel
from dl_model.old.resnet18.model import Resnet18
from dl_model.old.simplecnn.model import SimpleCNN
from dl_model.old.resnet50.model import Resnet50
from conv1d.model import CNNMLP
from dl_model.old.mlp.model1 import MLPModel1



# =========================
# 固定随机数
# =========================
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
def train(model):
    set_seed(42)

    base = Path(__file__).resolve().parents[1]
    data_path = base / "dl_model"/"mlp_feature_cache.jsonl"

    batch_size = 64
    lr = 3e-4
    epochs = 50

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("data_path :", data_path.resolve())
    print("train/test split from same file")
    print("device:", device)

    data = FeatureCacheDataset(data_path).records

    train_r, test_r = split_train_test(data)

    train_r = balance(train_r)
    test_r = balance(test_r)

    print("train size:", len(train_r))
    print("test size :", len(test_r))

    train_loader = DataLoader(PairDataset(train_r), batch_size, shuffle=True)
    test_loader = DataLoader(PairDataset(test_r), batch_size)

    model = model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", patience=3, factor=0.5, min_lr=1e-6
    )

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_test_acc = 0
    checkpoint_dir = base / "dl_model" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = checkpoint_dir / f"{model.__class__.__name__}_best.pth"

    final_train_loss = 0
    final_train_acc = 0
    final_test_loss = 0
    final_test_acc = 0

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

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save({
                "epoch": e + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
            }, best_model_path)
            print(f"  -> best model saved (test_acc={best_test_acc:.4f})")

        final_train_loss = train_loss
        final_train_acc = train_acc
        final_test_loss = test_loss
        final_test_acc = test_acc

    print("\n===== FINAL SUMMARY =====")
    print(f"train_loss: {final_train_loss:.4f}")
    print(f"train_acc : {final_train_acc:.4f}")
    print(f"test_loss : {final_test_loss:.4f}")
    print(f"test_acc  : {final_test_acc:.4f}")
    print(f"final_lr  : {opt.param_groups[0]['lr']:.6f}")

    final_model_path = checkpoint_dir / f"{model.__class__.__name__}_final.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "train_loss": final_train_loss,
        "train_acc": final_train_acc,
        "test_loss": final_test_loss,
        "test_acc": final_test_acc,
    }, final_model_path)
    print(f"final model saved: {final_model_path}")
    print(f"best model saved: {best_model_path}")


if __name__ == "__main__":
    #model = SimpleCNN()
    #model=Resnet18()
    #model=Resnet50()
    # model=MLPModel()
    # model=CNNMLP()
    model=MLPModel1()
    # model=MLPModel2()
    #model=TransformerModel()

    train(model)