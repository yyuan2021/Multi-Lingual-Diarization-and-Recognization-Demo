"""
使用训练好的 MLPModel1 模型评估测试集
不训练，直接推理

is_switch=True 表示同一个人说话（语码转换）
is_switch=False 表示不同人说话（mix）
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dl_model.old.mlp.model1 import MLPModel1
from dl_model.old.functions import SpeakerFeatureExtractor


def load_audio(path: Path, sr=16000):
    """加载音频"""
    wav, file_sr = torchaudio.load(str(path))
    if file_sr != sr:
        wav = torchaudio.functional.resample(wav, file_sr, sr)
    # 转单声道
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav.squeeze(0).cpu().numpy()  # [T]


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MLPModel1 on test set"
    )
    parser.add_argument(
        "--csv",
        default="dl_model/baseline_train_test_segments.csv",
        help="CSV file with labels"
    )
    parser.add_argument(
        "--test-dir",
        default="datasets/mlp_train/test",
        help="Test audio directory"
    )
    parser.add_argument(
        "--checkpoint",
        default="dl_model/checkpoints/MLPModel1_best.pth",
        help="Checkpoint to load"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=16000,
        help="Sample rate"
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=1.0,
        help="Window size on each side"
    )
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[1]
    root = base.parent
    
    csv_path = root / args.csv
    test_dir = root / args.test_dir
    checkpoint_path = root / args.checkpoint
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Test dir not found: {test_dir}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"test_dir: {test_dir}")
    print(f"checkpoint: {checkpoint_path}")

    # 加载测试样本
    print("\nLoading test samples...")
    test_samples = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if row["split"].lower() != "test":
                continue
            test_samples.append({
                "csv_index": idx + 1,
                "audio_path": row["audio_path"],
                "is_switch": row["is_switch"].lower() == "true",
                "switch_time": float(row["switch_time"]),
            })
    
    print(f"Test samples: {len(test_samples)}")

    # 加载模型
    print("\nLoading MLPModel1...")
    model = MLPModel1(input_dim=512).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print("Checkpoint loaded!")

    # 加载特征提取器
    extractor = SpeakerFeatureExtractor(sr=args.sr, model_name="base")

    # 预测
    print("\nPredicting...")
    all_preds = []
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for i in tqdm(range(0, len(test_samples), args.batch_size), desc="Processing", unit="batch"):
            batch_samples = test_samples[i:i + args.batch_size]

            batch_pairs = []
            batch_labels = []

            for sample in batch_samples:
                audio_file = test_dir / f"{sample['csv_index']}.wav"
                if not audio_file.exists():
                    tqdm.write(f"[WARN] Audio not found: {audio_file}")
                    continue

                # 加载完整 2 秒音频
                # 注意：音频文件已经是从 left_start 到 right_end 的 2 秒片段
                wav_full = load_audio(audio_file, sr=args.sr)

                # 音频文件已经是 [left_start, right_end] 的 2 秒
                # 所以直接从中间切开即可
                sr_int = int(args.sr)
                mid_point = len(wav_full) // 2
                
                left_wav = wav_full[:mid_point]
                right_wav = wav_full[mid_point:]
                
                # 填充到足够长度
                if len(left_wav) < sr_int:
                    left_wav = np.pad(left_wav, (0, sr_int - len(left_wav)), mode='constant')
                if len(right_wav) < sr_int:
                    right_wav = np.pad(right_wav, (0, sr_int - len(right_wav)), mode='constant')

                # 提取特征
                left_emb = extractor.extract_embedding(left_wav)
                right_emb = extractor.extract_embedding(right_wav)

                # 构建输入 [2, 512]
                pair = np.stack([left_emb, right_emb], axis=0)
                batch_pairs.append(pair)

                # is_switch=True -> label=1 (同一个人)
                # is_switch=False -> label=0 (不同人)
                batch_labels.append(1 if sample["is_switch"] else 0)

            if not batch_pairs:
                continue

            # 批处理
            batch_tensor = torch.from_numpy(np.stack(batch_pairs)).float().to(device)  # [B, 2, 512]

            # 预测
            logits = model(batch_tensor)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1).cpu().numpy()
            scores = probs[:, 1].cpu().numpy()  # is_switch 的概率

            all_preds.extend(preds)
            all_labels.extend(batch_labels)
            all_scores.extend(scores)
            
            # 实时显示当前准确率
            if len(all_labels) > 0:
                curr_acc = sum(1 for p, l in zip(all_preds, all_labels) if p == l) / len(all_labels)
                tqdm.write(f"  Batch {i//args.batch_size + 1}/{(len(test_samples)-1)//args.batch_size + 1}: "
                          f"curr_acc={curr_acc:.4f} ({len(all_labels)}/{len(test_samples)})")

    # 计算指标
    total = len(all_labels)
    correct = sum(1 for p, l in zip(all_preds, all_labels) if p == l)
    accuracy = correct / total if total > 0 else 0
    
    # TP/FP/FN/TN
    tp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 1)
    tn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 0)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # 打印结果
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS (MLPModel1)")
    print("=" * 60)
    print(f"Total samples: {total}")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("-" * 60)
    print("Confusion Matrix:")
    print(f"  TP (same→same): {tp}")
    print(f"  FP (diff→same): {fp}")
    print(f"  FN (same→diff): {fn}")
    print(f"  TN (diff→diff): {tn}")
    print("-" * 60)
    print(f"Score range: [{min(all_scores):.4f}, {max(all_scores):.4f}]")
    print("=" * 60)
    
    # 按 label 分组统计
    switch_total = sum(all_labels)  # is_switch=True 的数量
    switch_correct = sum(1 for p, l in zip(all_preds, all_labels) if p == l and l == 1)
    switch_acc = switch_correct / switch_total if switch_total > 0 else 0
    
    non_switch_total = total - switch_total
    non_switch_correct = sum(1 for p, l in zip(all_preds, all_labels) if p == l and l == 0)
    non_switch_acc = non_switch_correct / non_switch_total if non_switch_total > 0 else 0
    
    print("\nBreakdown by label:")
    print(f"  is_switch=True (same speaker):  {switch_acc:.4f} ({switch_correct}/{switch_total})")
    print(f"  is_switch=False (diff speaker): {non_switch_acc:.4f} ({non_switch_correct}/{non_switch_total})")
    print("=" * 60)


if __name__ == "__main__":
    main()
