import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dl_model.old.speechbrain.transfer import FrozenEncoderClassifier, SwitchDetectionDataset, evaluate, set_seed
from speechbrain.inference import EncoderClassifier


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SpeechBrain transfer model on a test-only dataset."
    )
    parser.add_argument(
        "--csv",
        default="dl_model/baseline_train_test_segments_switchlingua_seame.csv",
        help="CSV file with test labels",
    )
    parser.add_argument(
        "--test-audio-dir",
        default="datasets/baseline_switchlingua_seame_testset/test",
        help="Directory containing extracted test wav files",
    )
    parser.add_argument(
        "--checkpoint",
        default="dl_model/checkpoints/speechbrain_transfer_best.pth",
        help="Checkpoint to load",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    set_seed(args.seed)

    base = Path(__file__).resolve().parents[1]
    root = base.parent

    csv_path = root / args.csv
    test_audio_dir = root / args.test_audio_dir
    checkpoint_path = root / args.checkpoint

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not test_audio_dir.exists():
        raise FileNotFoundError(f"Test audio dir not found: {test_audio_dir}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if args.device == "auto":
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device=torch.device('cpu')
    else:
        device = torch.device(args.device)

    print(f"device: {device}")
    print(f"csv_path: {csv_path}")
    print(f"test_audio_dir: {test_audio_dir}")
    print(f"checkpoint: {checkpoint_path}")

    print("\nLoading SpeechBrain pretrained model...")
    encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device},
    )
    model = FrozenEncoderClassifier(encoder, embedding_dim=192, num_classes=2).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print("Checkpoint loaded!")

    print("\nLoading test dataset...")
    test_dataset = SwitchDetectionDataset(csv_path, test_audio_dir, sr=args.sr, split="test")
    print(f"Test samples: {len(test_dataset)}")
    if len(test_dataset) == 0:
        raise RuntimeError("No test samples found in CSV with split=test.")

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    loss_fn = torch.nn.CrossEntropyLoss()
    test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate(
        model, test_loader, device, loss_fn
    )

    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    print(f"test_loss : {test_loss:.4f}")
    print(f"test_acc  : {test_acc:.4f}")
    print(f"precision : {test_prec:.4f}")
    print(f"recall    : {test_rec:.4f}")
    print(f"f1        : {test_f1:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
