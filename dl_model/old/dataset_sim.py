# dataset_sim.py
import numpy as np
import torch
from torch.utils.data import Dataset

class SimSwitchDataset(Dataset):
    """
    模拟：
    - same speaker
    - diff speaker
    - 跨语言时某些特征扰动
    """

    def __init__(self, n_samples=5000):
        super().__init__()

        self.scalar_X = []
        self.mfcc_X = []
        self.y = []

        for _ in range(n_samples):

            same_speaker = np.random.rand() > 0.5
            cross_language = np.random.rand() > 0.5

            # --- 基础特征 ---
            # sim_rms, sim_energy_var, sim_pitch,
            # sim_zcr, sim_silence_ratio,
            # sim_duration, sim_time_gap, sim_mfcc_cos
            scalars = np.ones(8)

            if same_speaker:
                scalars *= np.random.normal(0.9, 0.05, 8)
            else:
                scalars *= np.random.normal(0.4, 0.1, 8)

            # 跨语言扰动：影响 pitch / energy
            if cross_language:
                scalars[2] *= np.random.normal(0.8, 0.1)  # pitch
                scalars[0] *= np.random.normal(0.7, 0.1)  # rms

            # MFCC per-dim 相似度
            if same_speaker:
                mfcc_dim = np.random.normal(0.95, 0.03, 13)
            else:
                mfcc_dim = np.random.normal(0.75, 0.08, 13)

            self.scalar_X.append(scalars)
            self.mfcc_X.append(mfcc_dim)
            self.y.append(int(same_speaker))

        self.scalar_X = torch.tensor(np.array(self.scalar_X), dtype=torch.float32)
        self.mfcc_X = torch.tensor(np.array(self.mfcc_X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.scalar_X[idx], self.mfcc_X[idx], self.y[idx]
