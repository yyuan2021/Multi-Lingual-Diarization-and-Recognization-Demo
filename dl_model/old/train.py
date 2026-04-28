# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

from dl_model.old.model import SwitchMLP
from dl_model.old.dataset_sim import SimSwitchDataset

def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SimSwitchDataset(n_samples=8000)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = SwitchMLP(scalar_dim=8, mfcc_dim=13).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(30):

        model.train()
        total_loss = 0
        preds = []
        labels = []

        for scalar_x, mfcc_x, y in train_loader:
            scalar_x = scalar_x.to(device)
            mfcc_x = mfcc_x.to(device)
            y = y.to(device)

            logits = model(scalar_x, mfcc_x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            preds.extend(torch.sigmoid(logits).detach().cpu().numpy())
            labels.extend(y.detach().cpu().numpy())

        preds = torch.tensor(preds).numpy().ravel()
        labels = torch.tensor(labels).numpy().ravel()

        auc = roc_auc_score(labels, preds)
        acc = accuracy_score(labels, (preds > 0.5).astype(int))

        print(f"Epoch {epoch+1:02d} | Loss {total_loss:.4f} | AUC {auc:.4f} | ACC {acc:.4f}")

    torch.save(model.state_dict(), "switch_mlp.pth")
    print("Model saved.")

if __name__ == "__main__":
    train()
