"""
使用预训练 SpeechBrain Speaker Verification 模型直接预测
不训练，直接推理

is_switch=True 表示同一个人说话（语码转换）
is_switch=False 表示不同人说话（mix）

每个音频文件 2 秒，中间 1 秒处是切换点
左段：[0s, 1s]  右段：[1s, 2s]
比较左右段的说话人相似度
"""

import argparse
import csv
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

from speechbrain.inference.speaker import SpeakerRecognition


def load_audio(path: Path, sr=16000):
    """加载音频"""
    wav, file_sr = torchaudio.load(str(path))
    if file_sr != sr:
        wav = torchaudio.functional.resample(wav, file_sr, sr)
    # 转单声道
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav  # [1, T]


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate pre-trained SpeechBrain speaker verification on test set"
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
        "--threshold",
        type=float,
        default=0.0,
        help="Similarity threshold (default: 0.0)"
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=16000,
        help="Sample rate"
    )
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[1]
    root = base.parent
    
    csv_path = root / args.csv
    test_dir = root / args.test_dir
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Test dir not found: {test_dir}")

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device=torch.device('cpu')
    print(f"device: {device}")
    print(f"test_dir: {test_dir}")
    print(f"threshold: {args.threshold}")

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
            })
    
    print(f"Test samples: {len(test_samples)}")

    # 加载预训练 Speaker Verification 模型
    print("\nLoading SpeechBrain Speaker Recognition model...")
    verification = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device}
    )

    # 预测
    print("\nPredicting...")
    all_preds = []
    all_labels = []
    all_scores = []  # 相似度分数

    with torch.no_grad():
        for sample in tqdm(test_samples, desc="Processing"):
            audio_file = test_dir / f"{sample['csv_index']}.wav"
            if not audio_file.exists():
                print(f"Warning: Audio not found: {audio_file}")
                continue
            
            # 加载完整 2 秒音频
            wav_full = load_audio(audio_file, sr=args.sr)  # [1, T]
            T = wav_full.shape[1]
            
            # 切分成左右段（各 1 秒）
            mid = T // 2
            wav_left = wav_full[:, :mid]  # [1, T/2]
            wav_right = wav_full[:, mid:]  # [1, T/2]
            
            # 确保长度一致
            min_len = min(wav_left.shape[1], wav_right.shape[1])
            wav_left = wav_left[:, :min_len]
            wav_right = wav_right[:, :min_len]
            
            # Speaker Verification：计算相似度
            score, prediction = verification.verify_batch(wav_left, wav_right)
            
            # score > 0 表示同一个人，score < 0 表示不同人
            score_float = float(score.cpu().numpy()[0])
            pred_label = 1 if score_float > args.threshold else 0
            
            # is_switch=True 表示同一个人 -> label=1
            # is_switch=False 表示不同人 -> label=0
            true_label = 1 if sample["is_switch"] else 0
            
            all_preds.append(pred_label)
            all_labels.append(true_label)
            all_scores.append(score_float)

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
    print("EVALUATION RESULTS (Pre-trained Speaker Verification)")
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
    print(f"Similarity threshold: {args.threshold}")
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
