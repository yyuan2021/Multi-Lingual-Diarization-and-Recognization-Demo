"""
评估模型在训练集和测试集上的 class 内精度
"""
import json
import random
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from dl_model.old.mlp.model1 import MLPModel1
from dl_model.old.simplecnn.model import SimpleCNN
from dl_model.old.resnet18.model import Resnet18
from dl_model.old.resnet50.model import Resnet50
from conv1d.model import CNNMLP
from dl_model.old.mlp.model import MLPModel
from dl_model.old.mlp.model2 import MLPModel2
from dl_model.old.transformer.model import TransformerModel


# =========================
# Dataset
# =========================
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


def evaluate_per_class(model, loader, device, class_name_map=None):
    """评估每个类别的精度"""
    model.eval()
    
    class_total = defaultdict(int)
    class_correct = defaultdict(int)
    
    total = 0
    correct = 0
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            
            logits = model(x)
            preds = logits.argmax(dim=1)
            
            for t, p in zip(y.tolist(), preds.tolist()):
                total += 1
                class_total[t] += 1
                if t == p:
                    correct += 1
                    class_correct[t] += 1
    
    # 整体精度
    overall_acc = correct / total if total > 0 else 0
    
    # 每个类别的精度
    per_class_acc = {}
    for cls in sorted(class_total.keys()):
        c_total = class_total[cls]
        c_correct = class_correct[cls]
        acc = c_correct / c_total if c_total > 0 else 0
        per_class_acc[cls] = {
            "total": c_total,
            "correct": c_correct,
            "accuracy": acc
        }
    
    return overall_acc, per_class_acc, class_total, class_correct


def load_model(model_class, checkpoint_path, device):
    """加载模型"""
    model = model_class().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


# =========================
# Main
# =========================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="模型 checkpoint 路径")
    parser.add_argument("--data", type=str, default="dl_model/mlp_feature_cache.jsonl", help="数据文件路径")
    parser.add_argument("--model", type=str, default="MLPModel1", 
                        choices=["MLPModel1", "MLPModel2", "MLPModel", "SimpleCNN", 
                                 "Resnet18", "Resnet50", "CNNMLP", "TransformerModel"],
                        help="模型类型")
    parser.add_argument("--batch-size", type=int, default=64, help="batch size")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--train-only", action="store_true", help="只评估训练集")
    parser.add_argument("--test-only", action="store_true", help="只评估测试集")
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    
    # 加载数据
    data_path = Path(args.data)
    print(f"data_path: {data_path.resolve()}")
    
    records = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except:
                continue
    
    print(f"total records: {len(records)}")
    
    # 划分训练/测试集（与训练时一致）
    train_records, test_records = split_train_test(records, seed=args.seed)
    print(f"train records: {len(train_records)}")
    print(f"test records: {len(test_records)}")
    
    # 加载模型
    model_class = globals()[args.model]
    model = load_model(model_class, args.checkpoint, device)
    print(f"model: {args.model}")
    print(f"checkpoint: {args.checkpoint}")
    
    # 创建 DataLoader
    batch_size = args.batch_size
    
    # 标签映射
    label_map = {
        0: "mix",           # is_switch=False
        1: "code_switch"    # is_switch=True
    }
    
    print("\n" + "=" * 60)
    
    # 评估训练集
    if not args.test_only:
        print("\n===== TRAIN SET =====")
        train_loader = DataLoader(PairDataset(train_records), batch_size=batch_size, shuffle=False)
        train_acc, train_per_class, train_total, train_correct = evaluate_per_class(
            model, train_loader, device
        )
        
        print(f"Overall Accuracy: {train_acc:.4f} ({train_correct}/{train_total})")
        print("\nPer-class Accuracy:")
        for cls in sorted(train_per_class.keys()):
            stats = train_per_class[cls]
            name = label_map.get(cls, str(cls))
            print(f"  {name:12s} (class {cls}): acc={stats['accuracy']:.4f} "
                  f"({stats['correct']}/{stats['total']})")
    
    # 评估测试集
    if not args.train_only:
        print("\n" + "=" * 60)
        print("\n===== TEST SET =====")
        test_loader = DataLoader(PairDataset(test_records), batch_size=batch_size, shuffle=False)
        test_acc, test_per_class, test_total, test_correct = evaluate_per_class(
            model, test_loader, device
        )
        
        print(f"Overall Accuracy: {test_acc:.4f} ({test_correct}/{test_total})")
        print("\nPer-class Accuracy:")
        for cls in sorted(test_per_class.keys()):
            stats = test_per_class[cls]
            name = label_map.get(cls, str(cls))
            print(f"  {name:12s} (class {cls}): acc={stats['accuracy']:.4f} "
                  f"({stats['correct']}/{stats['total']})")
    
    print("\n" + "=" * 60)
    
    # 输出混淆矩阵信息
    if not args.train_only and not args.train_only:
        print("\n===== CONFUSION MATRIX (TEST SET) =====")
        
        # 重新遍历测试集计算混淆矩阵
        model.eval()
        confusion = defaultdict(lambda: defaultdict(int))
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                preds = logits.argmax(dim=1)
                
                for t, p in zip(y.tolist(), preds.tolist()):
                    confusion[t][p] += 1
        
        print("                Pred")
        print("                mix(0)    code_sw(1)")
        print("True  mix(0)     {:8d}  {:8d}".format(
            confusion[0][0], confusion[0][1]))
        print("      code_sw(1) {:8d}  {:8d}".format(
            confusion[1][0], confusion[1][1]))


if __name__ == "__main__":
    main()
