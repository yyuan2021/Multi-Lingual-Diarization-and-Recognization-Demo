"""
SincNet Model Prediction Module for Same-Speaker Detection

Usage:
    1. Initialize model: model = SincNetPredictor(device="cpu", weight_path="xxx.pth")
    2. Call prediction: result = model.predict(audio1, audio2)
    3. Output: True (same speaker) / False (different speaker)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import librosa
import soundfile as sf

# Add project root to path for imports
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dl_model.old.speechbrain_ablation.shared import SincNetPairStudent, TDNNPairStudent


DEFAULT_SINCNET_WEIGHT = "dl_model/final_model/sincnet_best_acc.pth"
DEFAULT_TDNN_WEIGHT = "dl_model/final_model/tdnn_full_best_acc.pth"


class SincNetPredictor:
    """
    SincNet Same-Speaker Detection Predictor

    After loading the pretrained model, the predict method can be called
    repeatedly for inference.

    Attributes:
        device: Device to run inference on (cpu or cuda)
        model: Loaded TDNN model
        sample_rate: Audio sample rate (default 16000)
    """

    def __init__(
        self,
        device: str = "cpu",
        weight_path: str = DEFAULT_SINCNET_WEIGHT,
    ):
        """
        Initialize the predictor and load model weights.

        Args:
            device: Device to run on, default is "cpu"
            weight_path: Path to model weight file, default is sincnet_best_acc.pth
        """
        self.device = device
        self.sample_rate = 16000

        # Resolve weight path
        weight_path = Path(weight_path).expanduser()

        if not weight_path.is_absolute():
            repo_root = Path(__file__).resolve().parents[2]
            weight_path = repo_root / weight_path

        weight_path = Path(weight_path)
        
        if not weight_path.exists():
            raise FileNotFoundError(f"Weight file not found: {weight_path}")

        # Load checkpoint
        checkpoint = torch.load(weight_path, map_location=self.device, weights_only=False)
        saved_args = checkpoint.get("args", {})

        # Build model (pure SincNet)
        self.model = SincNetPairStudent(
            sample_rate=int(saved_args.get("sr", 16000)),
            emb_dim=int(saved_args.get("emb_dim", 192)),
            dropout=float(saved_args.get("dropout", 0.15)),
            sinc_channels=int(saved_args.get("sinc_channels", 80)),
        ).to(self.device)

        # Load weights and set to eval mode
        state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def _load_audio(self, audio_path: str) -> np.ndarray:
        """
        Load audio file and convert to mono 16kHz.

        Args:
            audio_path: Path to audio file

        Returns:
            Normalized audio waveform (numpy array)
        """
        wav, sr = sf.read(audio_path)

        # Convert to mono
        if wav.ndim > 1:
            wav = wav.mean(axis=1)

        # Resample to 16kHz if needed
        if sr != self.sample_rate:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sample_rate)

        return wav.astype(np.float32)

    def predict(self, audio1: np.ndarray, audio2: np.ndarray) -> Tuple[bool, float]:
        """
        Predict whether two audio segments are from the same speaker.

        Args:
            audio1: First audio waveform (numpy array, 16kHz)
            audio2: Second audio waveform (numpy array, 16kHz)

        Returns:
            (is_switch, confidence)
            is_switch: True if same speaker, False if different speaker
            confidence: Confidence score (probability between 0 and 1)
        """
        # Convert to tensor
        audio1_tensor = torch.tensor(audio1, dtype=torch.float32, device=self.device).unsqueeze(0)
        audio2_tensor = torch.tensor(audio2, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Inference
        with torch.no_grad():
            logits = self.model(audio1_tensor, audio2_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)

        # Get prediction result
        prediction = int(torch.argmax(probs).detach().cpu().item())
        confidence = float(probs[1].detach().cpu().item())  # Probability of class 1 (is_switch=True)

        is_switch = (prediction == 1)
        return is_switch, confidence


class TDNNPredictor:
    """TDNN Same-Speaker Detection Predictor using TDNNPairStudent."""

    def __init__(
        self,
        device: str = "cpu",
        weight_path: str = DEFAULT_TDNN_WEIGHT,
    ):
        self.device = device
        self.sample_rate = 16000

        weight_path = Path(weight_path).expanduser()
        if not weight_path.is_absolute():
            repo_root = Path(__file__).resolve().parents[2]
            weight_path = repo_root / weight_path
        if not weight_path.exists():
            raise FileNotFoundError(f"Weight file not found: {weight_path}")

        checkpoint = torch.load(weight_path, map_location=self.device, weights_only=False)
        saved_args = checkpoint.get("args", {})

        self.model = TDNNPairStudent(
            sample_rate=int(saved_args.get("sr", 16000)),
            n_mels=int(saved_args.get("n_mels", 40)),
            channels=tuple(saved_args.get("student_channels", [128, 192, 256, 256])),
            emb_dim=int(saved_args.get("emb_dim", 192)),
            dropout=float(saved_args.get("dropout", 0.15)),
            time_mask_max=int(saved_args.get("time_mask_max", 12)),
            freq_mask_max=int(saved_args.get("freq_mask_max", 6)),
            use_dilation=True,
            use_stats_pooling=True,
            use_pairwise_product=True,
            use_specaugment=True,
        ).to(self.device)

        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(self, audio1: np.ndarray, audio2: np.ndarray) -> Tuple[bool, float]:
        audio1_tensor = torch.tensor(audio1, dtype=torch.float32, device=self.device).unsqueeze(0)
        audio2_tensor = torch.tensor(audio2, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(audio1_tensor, audio2_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)
        prediction = int(torch.argmax(probs).detach().cpu().item())
        confidence = float(probs[1].detach().cpu().item())
        return (prediction == 1), confidence


if __name__ == "__main__":
    """
    Test script: Load a 2-second audio file, split it into two 1-second segments,
    and perform same-speaker prediction.

    Output explanation:
        is_switch=True  -> Both segments are from the SAME speaker
        is_switch=False -> Both segments are from DIFFERENT speakers
    """
    import argparse

    parser = argparse.ArgumentParser(description="Test SincNet Same-Speaker Detection Model")
    parser.add_argument(
        "--audio",
        type=str,
        default=r"dl_model\final_model\1.wav",
        help="Path to the 2-second audio file for testing"
    )
    parser.add_argument(
        "--weight",
        type=str,
        default=DEFAULT_SINCNET_WEIGHT.replace("/", "\\"),
        help="Path to model weight file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run inference on"
    )
    args = parser.parse_args()

    # Initialize predictor
    print(f"Loading model: {args.weight}")
    predictor = SincNetPredictor(device=args.device, weight_path=args.weight)
    print(f"Model loaded on {args.device}")

    # Load 2-second audio file
    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        exit(1)

    print(f"\nReading audio: {audio_path}")
    full_audio = predictor._load_audio(str(audio_path))
    print(f"Audio duration: {len(full_audio) / 16000:.4f} seconds")

    # Split 2-second audio into two 1-second segments
    mid_point = len(full_audio) // 2
    audio_first = full_audio[:mid_point]   # First 1 second
    audio_second = full_audio[mid_point:]  # Second 1 second

    print(f"\nFirst segment duration:  {len(audio_first) / 16000:.4f} seconds")
    print(f"Second segment duration: {len(audio_second) / 16000:.4f} seconds")

    # Call prediction
    print("\nPredicting...")
    is_switch, confidence = predictor.predict(audio_first, audio_second)

    # Output results
    print("\n" + "=" * 50)
    print("Prediction Result")
    print("=" * 50)
    print(f"is_switch (same speaker) = {is_switch}")
    print(f"confidence               = {confidence:.4f}")
    print("-" * 50)

    if is_switch:
        print("Conclusion: SAME SPEAKER")
    else:
        print("Conclusion: DIFFERENT SPEAKER")

    print("=" * 50)
