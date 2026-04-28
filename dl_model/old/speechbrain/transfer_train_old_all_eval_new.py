import argparse
import csv
import random
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from baseline.common import compute_metrics, load_eval_samples, preload_segment_pairs
from baseline.speechbrain_ecapa import SpeechBrainECAPABaseline
from speechbrain.inference import EncoderClassifier

from dl_model.old.speechbrain.transfer import FrozenEncoderClassifier, evaluate, set_seed


def parse_bool_label(value):
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes"}:
        return 1
    if text in {"0", "false", "f", "no"}:
        return 0
    raise ValueError(f"Unsupported is_switch value: {value}")


class OldAllTrainDataset(Dataset):
    """
    Use every row from the original baseline CSV for training, including rows whose
    original split was test. Audio is resolved from datasets/mlp_train/{train,test}.
    """

    def __init__(self, csv_path: Path, train_audio_dir: Path, test_audio_dir: Path, sr=16000):
        self.train_audio_dir = train_audio_dir
        self.test_audio_dir = test_audio_dir
        self.sr = sr
        self.samples = []

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                split = row["split"].strip().lower()
                audio_dir = self.train_audio_dir if split == "train" else self.test_audio_dir
                self.samples.append(
                    {
                        "audio_dir": audio_dir,
                        "csv_index": idx + 1,
                        "label": parse_bool_label(row["is_switch"]),
                    }
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        audio_file = sample["audio_dir"] / f"{sample['csv_index']}.wav"
        wav, sr = torchaudio.load(str(audio_file))
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        return wav.squeeze(0), sample["label"]


class IndexedEvalDataset(Dataset):
    """
    Evaluate on a CSV whose extracted wavs are named by test_row_index, falling back
    to the row order when that field is absent.
    """

    def __init__(self, csv_path: Path, audio_dir: Path, sr=16000, split="test"):
        self.audio_dir = audio_dir
        self.sr = sr
        self.samples = []

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                row_split = row.get("split", "").strip().lower()
                if split and row_split and row_split != split:
                    continue
                file_index = int(row["test_row_index"]) if row.get("test_row_index") else idx + 1
                self.samples.append(
                    {
                        "file_index": file_index,
                        "label": parse_bool_label(row["is_switch"]),
                    }
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        audio_file = self.audio_dir / f"{sample['file_index']}.wav"
        wav, sr = torchaudio.load(str(audio_file))
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        return wav.squeeze(0), sample["label"]


def evaluate_original_speechbrain_baseline(csv_path: Path, root: Path):
    samples, report = load_eval_samples(csv_path, root=root)
    pairs = preload_segment_pairs(samples, target_sr=16000)
    if not pairs:
        raise RuntimeError("No available rows found for original SpeechBrain baseline evaluation.")

    model = SpeechBrainECAPABaseline(device="cpu", cache_dir=root / "baseline" / "model_cache")
    labels = []
    predictions = []
    for pair in tqdm(pairs, desc="Original SpeechBrain baseline"):
        result = model.predict(pair.left_audio, pair.right_audio, pair.sample_rate)
        labels.append(pair.sample.label)
        predictions.append(int(result.prediction))

    metrics = compute_metrics(labels, predictions)
    print("\nOriginal SpeechBrain ECAPA on new testset")
    print(
        f"accuracy={metrics['accuracy']:.4f} "
        f"precision={metrics['precision']:.4f} "
        f"recall={metrics['recall']:.4f} "
        f"f1={metrics['f1']:.4f} "
        f"balanced_accuracy={metrics['balanced_accuracy']:.4f}"
    )
    print(
        f"positives={metrics['positives']} negatives={metrics['negatives']} "
        f"tp={metrics['tp']} tn={metrics['tn']} fp={metrics['fp']} fn={metrics['fn']}"
    )
    return metrics


def train(args):
    set_seed(args.seed)

    base = Path(__file__).resolve().parents[1]
    root = base.parent

    old_csv = root / args.old_csv
    old_train_audio_dir = root / args.old_train_audio_dir
    old_test_audio_dir = root / args.old_test_audio_dir
    new_test_csv = root / args.new_test_csv
    new_test_audio_dir = root / args.new_test_audio_dir

    for path in [old_csv, new_test_csv]:
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")
    for path in [old_train_audio_dir, old_test_audio_dir, new_test_audio_dir]:
        if not path.exists():
            raise FileNotFoundError(f"Audio dir not found: {path}")

    if args.device == "auto":
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device=torch.device('cpu')
    else:
        device = torch.device(args.device)
    print(f"device: {device}")
    print(f"old_csv: {old_csv}")
    print(f"old_train_audio_dir: {old_train_audio_dir}")
    print(f"old_test_audio_dir: {old_test_audio_dir}")
    print(f"new_test_csv: {new_test_csv}")
    print(f"new_test_audio_dir: {new_test_audio_dir}")

    original_baseline_metrics = evaluate_original_speechbrain_baseline(new_test_csv, root)

    print("\nLoading SpeechBrain pretrained model...")
    encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device},
    )

    model = FrozenEncoderClassifier(encoder, embedding_dim=192, num_classes=2).to(device)

    print("\nLoading datasets...")
    train_dataset = OldAllTrainDataset(
        old_csv,
        old_train_audio_dir,
        old_test_audio_dir,
        sr=args.sr,
    )
    test_dataset = IndexedEvalDataset(
        new_test_csv,
        new_test_audio_dir,
        sr=args.sr,
        split="test",
    )

    print(f"Train samples (old train + old test): {len(train_dataset)}")
    print(f"Test samples (new testset): {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    optimizer = torch.optim.Adam(
        model.classifier.parameters(),
        lr=args.lr,
        weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=3,
        factor=0.5,
        min_lr=1e-6,
    )
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    checkpoint_dir = root / args.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = checkpoint_dir / args.best_name
    final_model_path = checkpoint_dir / args.final_name

    print("\nStarting training...")
    print(f"batch_size: {args.batch_size}")
    print(f"lr: {args.lr}")
    print(f"epochs: {args.epochs}")
    print(f"early_stopping_patience: {args.patience}")
    print("-" * 70)

    best_test_acc = 0.0
    best_test_f1 = 0.0
    best_epoch = 0
    epochs_without_improvement = 0

    default_test_loss, default_test_acc, default_test_prec, default_test_rec, default_test_f1 = evaluate(
        model, test_loader, device, loss_fn
    )
    print(
        "Default model before training | "
        f"test_loss={default_test_loss:.4f} test_acc={default_test_acc:.4f} "
        f"precision={default_test_prec:.4f} recall={default_test_rec:.4f} "
        f"f1={default_test_f1:.4f}"
    )

    for epoch in range(args.epochs):
        model.train()
        train_loss_sum = 0.0
        train_total = 0
        train_correct = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for wav, labels in pbar:
            wav, labels = wav.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(wav)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * wav.size(0)
            train_total += wav.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{train_correct / train_total:.4f}",
            )

        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total

        test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate(
            model, test_loader, device, loss_fn
        )
        scheduler.step(test_loss)

        print(
            f"Epoch {epoch + 1:02d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} "
            f"precision={test_prec:.4f} recall={test_rec:.4f} f1={test_f1:.4f}"
        )

        improved = test_acc > best_test_acc or (
            abs(test_acc - best_test_acc) < 1e-8 and test_f1 > best_test_f1
        )
        if improved:
            best_test_acc = test_acc
            best_test_f1 = test_f1
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "test_acc": test_acc,
                    "test_f1": test_f1,
                    "args": vars(args),
                },
                best_model_path,
            )
            print(f"Saved best checkpoint to: {best_model_path}")
        else:
            epochs_without_improvement += 1
            print(
                f"No improvement for {epochs_without_improvement} epoch(s). "
                f"Best so far: epoch {best_epoch}, acc={best_test_acc:.4f}, f1={best_test_f1:.4f}"
            )
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping triggered at epoch {epoch + 1}.")
                break

    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_test_acc": best_test_acc,
            "best_test_f1": best_test_f1,
            "best_epoch": best_epoch,
            "args": vars(args),
        },
        final_model_path,
    )

    print("\nTraining finished.")
    print(f"Original SpeechBrain baseline acc: {original_baseline_metrics['accuracy']:.4f}")
    print(f"Default classifier-head test acc: {default_test_acc:.4f}")
    print(f"Best test acc: {best_test_acc:.4f}")
    print(f"Best test f1 : {best_test_f1:.4f}")
    print(f"Best epoch   : {best_epoch}")
    print(f"Best checkpoint : {best_model_path}")
    print(f"Final checkpoint: {final_model_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train SpeechBrain transfer model on all original samples and evaluate on the new test set."
    )
    parser.add_argument(
        "--old-csv",
        default="dl_model/baseline_train_test_segments.csv",
    )
    parser.add_argument(
        "--old-train-audio-dir",
        default="datasets/mlp_train/train",
    )
    parser.add_argument(
        "--old-test-audio-dir",
        default="datasets/mlp_train/test",
    )
    parser.add_argument(
        "--new-test-csv",
        default="dl_model/baseline_train_test_segments_switchlingua_seame.csv",
    )
    parser.add_argument(
        "--new-test-audio-dir",
        default="datasets/baseline_switchlingua_seame_testset/test",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="dl_model/checkpoints",
    )
    parser.add_argument(
        "--best-name",
        default="speechbrain_transfer_old_all_to_switchlingua_seame_best.pth",
    )
    parser.add_argument(
        "--final-name",
        default="speechbrain_transfer_old_all_to_switchlingua_seame_final.pth",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
