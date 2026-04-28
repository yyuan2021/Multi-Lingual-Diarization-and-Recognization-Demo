# model.py
import torch
import torch.nn as nn

class SwitchMLP(nn.Module):
    def __init__(self, scalar_dim=8, mfcc_dim=13):
        super().__init__()

        self.scalar_branch = nn.Sequential(
            nn.Linear(scalar_dim, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        self.mfcc_branch = nn.Sequential(
            nn.Linear(mfcc_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.fusion_head = nn.Sequential(
            nn.Linear(16 + 32, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, scalar_x, mfcc_x):
        scalar_feat = self.scalar_branch(scalar_x)
        mfcc_feat = self.mfcc_branch(mfcc_x)

        fused = torch.cat([scalar_feat, mfcc_feat], dim=1)
        return self.fusion_head(fused)
