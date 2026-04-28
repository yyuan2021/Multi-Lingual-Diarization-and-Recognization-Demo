import torch
import torch.nn as nn


# =========================
# Encoder (Conv1D + MLP)
# =========================
class CNNMLPEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, 3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)   # → (B,256,1)
        )

        self.mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

    def forward(self, x):
        """
        x: (B, 1, D)
        """
        x = self.conv(x).squeeze(-1)   # (B,256)
        x = self.mlp(x)                # (B,256)
        return x


# =========================
# Main Model
# =========================
class CNNMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = CNNMLPEncoder()

        self.fc = nn.Sequential(
            nn.Linear(256 * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        """
        x: (B, 2, D)
        """
        l = self.enc(x[:, 0].unsqueeze(1))
        r = self.enc(x[:, 1].unsqueeze(1))
        d = torch.abs(l - r)
        return self.fc(torch.cat([l, r, d], dim=1))