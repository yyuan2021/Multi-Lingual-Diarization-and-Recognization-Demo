"""
使用 SpeechBrain 预训练 speaker recognition 模型进行迁移学习
冻结编码器层，训练分类头判断是否发生语码转换（code-switching）

数据集格式：
- datasets/mlp_train/train/ 和 datasets/mlp_train/test/ 存放 2 秒音频片段
- baseline_train_test_segments.csv 提供标注（is_switch 列）
"""

import argparse
import csv
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
from tqdm import tqdm

from speechbrain.inference import EncoderClassifier


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
class SwitchDetectionDataset(Dataset):
    """
    从 CSV 读取标注，加载对应的音频片段
    CSV 格式：audio_path,is_switch,split,left_start,switch_time,right_end
    音频文件：datasets/mlp_train/{train,test}/{row_index}.wav
    """
    def __init__(self, csv_path: Path, audio_dir: Path, sr=16000, split: str = "train"):
        self.audio_dir = audio_dir
        self.sr = sr
        self.samples = []

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                # 只保留匹配的 split
                if row["split"].lower() != split:
                    continue
                self.samples.append({
                    "audio_path": row["audio_path"],
                    "is_switch": row["is_switch"].lower() == "true",
                    "split": row["split"].lower(),
                    "switch_time": float(row["switch_time"]),
                    "csv_index": idx + 1,  # 文件名对应 CSV 行号（从 1 开始）
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 音频文件名：{csv_index}.wav
        audio_file = self.audio_dir / f"{sample['csv_index']}.wav"
        
        # 加载音频
        wav, sr = torchaudio.load(str(audio_file))
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        
        # 转为单声道
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        
        # 去掉 batch 维度 [1, T]
        wav = wav.squeeze(0)
        
        # 标签：is_switch -> 1, not switch -> 0
        label = 1 if sample["is_switch"] else 0

        return wav, label


# =========================
# 模型：冻结编码器 + 可训练分类头
# =========================
class FrozenEncoderClassifier(nn.Module):
    """
    冻结 SpeechBrain 编码器，只训练最后的分类层
    """
    def __init__(self, encoder, embedding_dim=192, num_classes=2):
        super().__init__()
        self.encoder = encoder
        self.embedding_dim = embedding_dim
        
        # 冻结编码器所有参数
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # 可训练的分类头
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, wav):
        """
        wav: [B, T] 原始波形
        返回：[B, num_classes] logits
        """
        with torch.no_grad():
            # 使用 SpeechBrain 的 encode_batch 获取 embedding
            embeddings = self.encoder.encode_batch(wav)
            # embeddings: [B, 1, embedding_dim] -> [B, embedding_dim]
            embeddings = embeddings.squeeze(1)
        
        logits = self.classifier(embeddings)
        return logits


# =========================
# 训练/评估
# =========================
def evaluate(model, loader, device, loss_fn):
    model.eval()
    total_loss = 0
    total = 0
    correct = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for wav, labels in loader:
            wav, labels = wav.to(device), labels.to(device)
            
            logits = model(wav)
            loss = loss_fn(logits, labels)
            
            total_loss += loss.item() * wav.size(0)
            total += wav.size(0)
            
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = correct / total
    
    # 计算 precision/recall/f1
    tp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 1)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return total_loss / total, acc, precision, recall, f1


def train(args):
    set_seed(args.seed)

    base = Path(__file__).resolve().parents[1]
    root = base.parent
    
    # 数据路径
    csv_path = root / "dl_model" / "baseline_train_test_segments.csv"
    train_audio_dir = root / "datasets" / "mlp_train" / "train"
    test_audio_dir = root / "datasets" / "mlp_train" / "test"
    
    # 检查文件
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not train_audio_dir.exists():
        raise FileNotFoundError(f"Train audio dir not found: {train_audio_dir}")
    if not test_audio_dir.exists():
        raise FileNotFoundError(f"Test audio dir not found: {test_audio_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"csv_path: {csv_path}")
    print(f"train_audio_dir: {train_audio_dir}")
    print(f"test_audio_dir: {test_audio_dir}")

    # 加载预训练 SpeechBrain 模型
    print("\nLoading SpeechBrain pretrained model...")
    encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device}
    )
    embedding_dim = 192  # ECAPA-TDNN 的 embedding 维度

    # 创建模型
    model = FrozenEncoderClassifier(encoder, embedding_dim=embedding_dim, num_classes=2)
    model = model.to(device)

    # 创建数据集
    print("\nLoading datasets...")
    train_dataset = SwitchDetectionDataset(csv_path, train_audio_dir, sr=16000, split="train")
    test_dataset = SwitchDetectionDataset(csv_path, test_audio_dir, sr=16000, split="test")
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # 数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 优化器（只优化分类头）
    optimizer = torch.optim.Adam(
        model.classifier.parameters(), 
        lr=args.lr,
        weight_decay=1e-5
    )

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5, min_lr=1e-6
    )

    # 损失函数
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    # 检查点目录
    checkpoint_dir = base / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = checkpoint_dir / "speechbrain_transfer_best.pth"
    final_model_path = checkpoint_dir / "speechbrain_transfer_final.pth"

    # 训练循环
    print("\nStarting training...")
    print(f"batch_size: {args.batch_size}")
    print(f"lr: {args.lr}")
    print(f"epochs: {args.epochs}")
    print("-" * 70)

    best_test_acc = 0
    best_test_f1 = 0

    for epoch in range(args.epochs):
        # 训练
        model.train()
        train_loss_sum = 0
        train_total = 0
        train_correct = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for wav, labels in pbar:
            wav, labels = wav.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(wav)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.classifier.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item() * wav.size(0)
            train_total += wav.size(0)

            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()

            # 进度条显示 loss 和 acc
            curr_train_acc = train_correct / train_total if train_total > 0 else 0
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "train_acc": f"{curr_train_acc:.4f}"
            })

        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total

        # 评估
        test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate(
            model, test_loader, device, loss_fn
        )

        scheduler.step(train_loss)

        # 打印进度
        print(
            f"{epoch+1:03d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} | "
            f"P={test_prec:.4f} R={test_rec:.4f} F1={test_f1:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.6f}"
        )

        # 保存最佳模型（按 F1）
        if test_f1 > best_test_f1:
            best_test_f1 = test_f1
            best_test_acc = test_acc
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "classifier_state_dict": model.classifier.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "test_precision": test_prec,
                "test_recall": test_rec,
                "test_f1": test_f1,
            }, best_model_path)
            print(f"  -> best model saved (test_f1={test_f1:.4f})")

    # 最终总结
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"best_test_acc : {best_test_acc:.4f}")
    print(f"best_test_f1  : {best_test_f1:.4f}")
    print(f"final_lr      : {optimizer.param_groups[0]['lr']:.6f}")
    print(f"best_model    : {best_model_path}")
    print(f"final_model   : {final_model_path}")

    # 保存最终模型
    torch.save({
        "model_state_dict": model.state_dict(),
        "classifier_state_dict": model.classifier.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, final_model_path)
    
    print("\nTraining completed!")


def main():
    parser = argparse.ArgumentParser(
        description="Transfer learning with SpeechBrain for code-switching detection"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
