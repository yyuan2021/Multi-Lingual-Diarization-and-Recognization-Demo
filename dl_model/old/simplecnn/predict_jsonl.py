import argparse
import json
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# =========================
# Model（必须和训练一致）
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
# 主流程
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default="result.csv")
    args = parser.parse_args()

    input_path = Path(args.input)
    model_path = Path(args.model)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ===== load model =====
    model = SwitchCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    results = []
    total = 0
    correct = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
            except:
                continue

            # ===== 取 embedding =====
            emb1 = np.array(data["feature"]["left"]["embedding"], dtype=np.float32)
            emb2 = np.array(data["feature"]["right"]["embedding"], dtype=np.float32)

            pair = np.stack([emb1, emb2], axis=0)
            x = torch.from_numpy(pair).unsqueeze(0).to(device)

            # ===== 预测 =====
            with torch.no_grad():
                logits = model(x)
                pred = torch.argmax(logits, dim=1).item()

            true = int(data["is_switch"])

            if pred == true:
                correct += 1
            total += 1

            results.append([
                data.get("audio_path", ""),
                true,
                pred
            ])

    acc = correct / max(total, 1)
    print(f"Accuracy: {acc:.4f} ({correct}/{total})")

    # ===== 写 CSV =====
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["audio_path", "true_label", "pred_label"])
        writer.writerows(results)

    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()