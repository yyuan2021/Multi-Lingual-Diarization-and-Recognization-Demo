import argparse
import csv
import importlib
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dl_model.old.speechbrain_ablation.shared import (
    DistillationPairDataset,
    augment_waveforms,
    benchmark_student,
    build_samples_from_new_extracted,
    build_samples_from_old_all,
    collate_audio_pairs,
    evaluate_student,
    load_soft_labels,
    set_seed,
    soft_distill_loss,
)


MODEL_MODULES = {
    "pure_sincnet": "dl_model.speechbrain_ablation.model_pure_sincnet",
    "sincnet": "dl_model.speechbrain_ablation.model_sincnet",
    "tdnn_full": "dl_model.speechbrain_ablation.model_tdnn_full",
    "no_dilation": "dl_model.speechbrain_ablation.model_no_dilation",
    "no_stats_pooling": "dl_model.speechbrain_ablation.model_no_stats_pooling",
    "no_pairwise_product": "dl_model.speechbrain_ablation.model_no_pairwise_product",
    "no_specaugment": "dl_model.speechbrain_ablation.model_no_specaugment",
    "embedding_classifier": "dl_model.speechbrain_ablation.model_embedding_classifier",
    "mlp": "dl_model.speechbrain_ablation.model_mlp",
    "cnn": "dl_model.speechbrain_ablation.model_cnn",
    "transformer": "dl_model.speechbrain_ablation.model_transformer",
    "resnet": "dl_model.speechbrain_ablation.model_resnet",
}


def import_builder(model_name):
    module = importlib.import_module(MODEL_MODULES[model_name])
    return module.build_model


def mean_std(values):
    if not values:
        return None, None
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, math.sqrt(var)


def train_one_model(model_name, args, train_samples, test_samples, device, run_index):
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_run{run_index}" if args.repeat > 1 else ""
    best_acc_path = checkpoint_dir / f"{model_name}{suffix}_best_acc.pth"
    best_f1_path = checkpoint_dir / f"{model_name}{suffix}_best_f1.pth"
    final_path = checkpoint_dir / f"{model_name}{suffix}_final.pth"

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

    model = import_builder(model_name)(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr,
    )
    ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.03)
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_f1 = -1.0
    best_f1_acc = 0.0
    best_f1_epoch = 0
    best_acc = -1.0
    best_acc_f1 = -1.0
    best_acc_epoch = 0
    no_improve = 0

    best_acc_record = None
    best_f1_record = None

    print(f"\n=== Training {model_name} (run {run_index}/{args.repeat}) ===")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        pbar = tqdm(train_loader, desc=f"{model_name} {epoch}/{args.epochs}")
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
                hard_loss = ce_loss_fn(logits, labels)
                distill_loss = soft_distill_loss(logits, teacher_prob, args.temperature)
                loss = args.alpha * hard_loss + (1.0 - args.alpha) * distill_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item() * labels.size(0)
            train_count += labels.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = train_loss_sum / train_count
        test_metrics = evaluate_student(model, test_loader, device, ce_loss_fn, use_tta_swap=args.eval_tta_swap)
        scheduler.step()

        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} "
            f"test_acc={test_metrics['accuracy']:.4f} test_f1={test_metrics['f1']:.4f} "
            f"test_loss={test_metrics['loss']:.4f}"
        )

        current_acc = test_metrics["accuracy"] if test_metrics["accuracy"] is not None else -1.0
        current_f1 = test_metrics["f1"] if test_metrics["f1"] is not None else -1.0

        improved_acc = current_acc > best_acc or (
            abs(current_acc - best_acc) < 1e-8 and current_f1 > best_acc_f1
        )
        if improved_acc:
            best_acc = current_acc
            best_acc_f1 = current_f1
            best_acc_epoch = epoch
            best_acc_record = {
                "model": model_name,
                "run": run_index,
                "epoch": epoch,
                "test_acc": float(current_acc),
                "test_f1": float(current_f1),
                "test_loss": float(test_metrics["loss"]),
                "test_precision": float(test_metrics["precision"]),
                "test_recall": float(test_metrics["recall"]),
                "train_loss": float(train_loss),
                "args": vars(args),
            }
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "test_metrics": test_metrics,
                    "args": vars(args),
                    "model_name": model_name,
                    "selection_metric": "accuracy",
                },
                best_acc_path,
            )

        improved_f1 = current_f1 > best_f1 or (
            abs(current_f1 - best_f1) < 1e-8 and current_acc > best_f1_acc
        )
        if improved_f1:
            best_f1 = current_f1
            best_f1_acc = current_acc
            best_f1_epoch = epoch
            no_improve = 0
            best_f1_record = {
                "model": model_name,
                "run": run_index,
                "epoch": epoch,
                "test_acc": float(current_acc),
                "test_f1": float(current_f1),
                "test_loss": float(test_metrics["loss"]),
                "test_precision": float(test_metrics["precision"]),
                "test_recall": float(test_metrics["recall"]),
                "train_loss": float(train_loss),
                "args": vars(args),
            }
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "test_metrics": test_metrics,
                    "args": vars(args),
                    "model_name": model_name,
                    "selection_metric": "f1",
                },
                best_f1_path,
            )
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stopping for {model_name} at epoch {epoch}.")
                break

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "best_acc_epoch": best_acc_epoch,
            "best_f1_epoch": best_f1_epoch,
            "best_acc": best_acc,
            "best_f1": best_f1,
            "best_acc_f1": best_acc_f1,
            "best_f1_acc": best_f1_acc,
            "args": vars(args),
            "model_name": model_name,
        },
        final_path,
    )

    # Save best acc and best f1 records to JSON files
    suffix = f"_run{run_index}" if args.repeat > 1 else ""
    best_acc_record_path = checkpoint_dir / f"{model_name}{suffix}_best_acc_record.json"
    best_f1_record_path = checkpoint_dir / f"{model_name}{suffix}_best_f1_record.json"

    if best_acc_record is not None:
        with open(best_acc_record_path, "w", encoding="utf-8") as f:
            json.dump(best_acc_record, f, indent=2, ensure_ascii=False)

    if best_f1_record is not None:
        with open(best_f1_record_path, "w", encoding="utf-8") as f:
            json.dump(best_f1_record, f, indent=2, ensure_ascii=False)

    student_ms = benchmark_student(model, test_dataset, device=device, limit=args.benchmark_samples)
    return {
        "model": model_name,
        "run": run_index,
        "best_acc_epoch": best_acc_epoch,
        "best_f1_epoch": best_f1_epoch,
        "best_acc": best_acc,
        "best_f1": best_f1,
        "best_acc_f1": best_acc_f1,
        "best_f1_acc": best_f1_acc,
        "student_ms": student_ms,
        "best_acc_checkpoint": str(best_acc_path),
        "best_f1_checkpoint": str(best_f1_path),
        "final_checkpoint": str(final_path),
        "best_acc_record": str(best_acc_record_path) if best_acc_record else None,
        "best_f1_record": str(best_f1_record_path) if best_f1_record else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Unified ablation/comparison trainer for distilled speaker models.")
    parser.add_argument("--models", nargs="+", default=list(MODEL_MODULES.keys()), choices=list(MODEL_MODULES.keys()))
    parser.add_argument("--old-csv", default="dl_model/baseline_train_test_segments.csv")
    parser.add_argument("--old-train-audio-dir", default="datasets/mlp_train/train")
    parser.add_argument("--old-test-audio-dir", default="datasets/mlp_train/test")
    parser.add_argument("--new-test-csv", default="dl_model/baseline_train_test_segments_switchlingua_seame.csv")
    parser.add_argument("--new-test-audio-dir", default="datasets/baseline_switchlingua_seame_testset/test")
    parser.add_argument("--soft-labels-cache", default="dl_model/checkpoints/speechbrain_soft_labels_old_all_eval_new.pt")
    parser.add_argument("--checkpoint-dir", default="dl_model/speechbrain_ablation/checkpoints")
    parser.add_argument("--summary-path", default="dl_model/speechbrain_ablation/summary.json")
    parser.add_argument("--student-device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--temperature", type=float, default=2.0)
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
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    train_samples = build_samples_from_old_all(
        root / args.old_csv,
        root / args.old_train_audio_dir,
        root / args.old_test_audio_dir,
        target_sr=args.sr,
    )
    test_samples = build_samples_from_new_extracted(
        root / args.new_test_csv,
        root / args.new_test_audio_dir,
        target_sr=args.sr,
        split="test",
    )
    load_soft_labels(root / args.soft_labels_cache, train_samples, test_samples)

    if args.student_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.student_device)
    print(f"Using device: {device}")
    print(f"Train samples: {len(train_samples)} | Test samples: {len(test_samples)}")

    summary_rows = []
    aggregate_rows = []
    for model_name in args.models:
        model_runs = []
        for run_index in range(1, args.repeat + 1):
            set_seed(args.seed + run_index - 1)
            result = train_one_model(model_name, args, train_samples, test_samples, device, run_index)
            summary_rows.append(result)
            model_runs.append(result)

        acc_mean, acc_std = mean_std([row["best_acc"] for row in model_runs])
        f1_mean, f1_std = mean_std([row["best_f1"] for row in model_runs])
        ms_values = [row["student_ms"] for row in model_runs if row["student_ms"] is not None]
        ms_mean, ms_std = mean_std(ms_values)
        aggregate_rows.append(
            {
                "model": model_name,
                "runs": len(model_runs),
                "best_acc_mean": acc_mean,
                "best_acc_std": acc_std,
                "best_f1_mean": f1_mean,
                "best_f1_std": f1_std,
                "student_ms_mean": ms_mean,
                "student_ms_std": ms_std,
            }
        )

    summary_path = root / args.summary_path
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "runs": summary_rows,
                "aggregate": aggregate_rows,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    csv_path = summary_path.with_suffix(".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    aggregate_csv_path = summary_path.with_name(summary_path.stem + "_aggregate.csv")
    with open(aggregate_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregate_rows[0].keys()))
        writer.writeheader()
        writer.writerows(aggregate_rows)

    # Save best acc and best f1 records to CSV
    best_acc_rows = [row for row in summary_rows if row.get("best_acc_record")]
    best_f1_rows = [row for row in summary_rows if row.get("best_f1_record")]

    if best_acc_rows:
        best_acc_csv_path = summary_path.with_name(summary_path.stem + "_best_acc.csv")
        best_acc_data = []
        for row in best_acc_rows:
            record_path = Path(row["best_acc_record"])
            if record_path.exists():
                with open(record_path, "r", encoding="utf-8") as f:
                    record = json.load(f)
                best_acc_data.append(record)
        if best_acc_data:
            with open(best_acc_csv_path, "w", newline="", encoding="utf-8") as f:
                fieldnames = ["model", "run", "epoch", "test_acc", "test_f1", "test_loss", "test_precision", "test_recall", "train_loss"]
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(best_acc_data)

    if best_f1_rows:
        best_f1_csv_path = summary_path.with_name(summary_path.stem + "_best_f1.csv")
        best_f1_data = []
        for row in best_f1_rows:
            record_path = Path(row["best_f1_record"])
            if record_path.exists():
                with open(record_path, "r", encoding="utf-8") as f:
                    record = json.load(f)
                best_f1_data.append(record)
        if best_f1_data:
            with open(best_f1_csv_path, "w", newline="", encoding="utf-8") as f:
                fieldnames = ["model", "run", "epoch", "test_acc", "test_f1", "test_loss", "test_precision", "test_recall", "train_loss"]
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(best_f1_data)

    print(f"\nSaved summary to: {summary_path}")
    print(f"Saved summary csv to: {csv_path}")
    print(f"Saved aggregate csv to: {aggregate_csv_path}")
    if best_acc_rows:
        print(f"Saved best acc records csv to: {best_acc_csv_path}")
    if best_f1_rows:
        print(f"Saved best f1 records csv to: {best_f1_csv_path}")


if __name__ == "__main__":
    main()
