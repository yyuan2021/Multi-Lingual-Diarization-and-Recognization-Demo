import torch
import torch.nn as nn
import torchvision.models as models


# =========================
# Encoder (ResNet18)
# =========================
class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = models.resnet18(weights=None)

        # 🔴 输入改成 1 channel
        backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # 🔴 去掉最后 fc
        self.feature = nn.Sequential(*list(backbone.children())[:-1])  # (B,512,1,1)

    def forward(self, x):
        """
        x: (B, 1, D)
        """
        x = x.unsqueeze(-1)          # (B,1,D,1)
        x = self.feature(x)          # (B,512,1,1)
        return x.view(x.size(0), -1) # (B,512)


# =========================
# Main Model（不变逻辑）
# =========================
class Resnet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = CNNEncoder()
        self.fc = nn.Sequential(
            nn.Linear(512 * 3, 128),   # 🔴 改成 512
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