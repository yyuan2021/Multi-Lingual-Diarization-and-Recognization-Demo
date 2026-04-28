import torch
import torch.nn as nn


# =========================
# Encoder (Transformer)
# =========================
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim=512, seq_len=32, d_model=64, nhead=4, num_layers=2):
        super().__init__()

        self.seq_len = seq_len
        self.chunk_dim = input_dim // seq_len  # 512 / 32 = 16

        self.input_proj = nn.Linear(self.chunk_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        """
        x: (B, 1, 512)
        """
        x = x.squeeze(1)                          # (B,512)

        # 🔴 reshape 成 sequence
        x = x.view(x.size(0), self.seq_len, -1)   # (B,32,16)

        x = self.input_proj(x)                    # (B,32,64)

        x = self.encoder(x)                       # (B,32,64)

        x = x.transpose(1, 2)                     # (B,64,32)
        x = self.pool(x).squeeze(-1)              # (B,64)

        return x


# =========================
# Main Model
# =========================
class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc = TransformerEncoder()

        self.fc = nn.Sequential(
            nn.Linear(64 * 4, 128),   # l, r, diff, prod
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        """
        x: (B, 2, 512)
        """
        l = self.enc(x[:, 0].unsqueeze(1))
        r = self.enc(x[:, 1].unsqueeze(1))

        diff = torch.abs(l - r)
        prod = l * r

        return self.fc(torch.cat([l, r, diff, prod], dim=1))