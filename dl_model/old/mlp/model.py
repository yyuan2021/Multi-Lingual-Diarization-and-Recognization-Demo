import torch
import torch.nn as nn


# =========================
# Encoder (MLP)
# =========================
class MLPEncoder(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        """
        x: (B, 1, D)
        """
        x = x.squeeze(1)          # (B, D)
        return self.net(x)        # (B, hidden_dim)


# =========================
# Main Model
# =========================
class MLPModel(nn.Module):
    def __init__(self, input_dim=512):
        super().__init__()
        self.enc = MLPEncoder(input_dim=input_dim, hidden_dim=256)

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