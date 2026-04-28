import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Residual Block
# =========================
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.net(x))


# =========================
# Encoder
# =========================
class MLPEncoder(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256):
        super().__init__()

        self.input = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        self.res1 = ResidualBlock(hidden_dim)
        self.res2 = ResidualBlock(hidden_dim)

    def forward(self, x):
        """
        x: (B, 1, D)
        """
        x = x.squeeze(1)
        x = self.input(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


# =========================
# Main Model（只改这里的interaction）
# =========================
class MLPModel2(nn.Module):
    def __init__(self, input_dim=512):
        super().__init__()

        self.enc = MLPEncoder(input_dim=input_dim, hidden_dim=256)

        self.fc = nn.Sequential(
            nn.Linear(256 * 4 + 1, 128),  # 🔴 多了 cosine → +1
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

        diff = torch.abs(l - r)
        prod = l * r

        # 🔴 新增：cosine similarity
        cos = F.cosine_similarity(l, r, dim=1).unsqueeze(1)

        return self.fc(torch.cat([l, r, diff, prod, cos], dim=1))