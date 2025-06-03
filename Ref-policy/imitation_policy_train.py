import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split

class PendulumImitationDataset(Dataset):
    def __init__(self, csv_path, embedding_dir, indices=None):
        self.df = pd.read_csv(csv_path)
        self.embedding_dir = embedding_dir
        self.trajs = self._split_trajectories()
        self.samples = self._make_samples()
        if indices is not None:
            self.samples = [self.samples[i] for i in indices]

    def _split_trajectories(self):
        trajs = []
        for start in range(0, len(self.df), 200):
            trajs.append(self.df.iloc[start:start+200].reset_index(drop=True))
        return trajs

    def _make_samples(self):
        samples = []
        for traj in self.trajs:
            for i in range(len(traj)):
                samples.append((traj, i))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        traj, i = self.samples[idx]
        z = np.load(os.path.join(self.embedding_dir, traj.loc[i, 'state_image'].replace('.png', '.npy')))
        action = np.array(eval(traj.loc[i, 'action']), dtype=np.float32)
        return {
            'z': torch.tensor(z, dtype=torch.float32), # (N, E)
            'action': torch.tensor(action, dtype=torch.float32), # (action_dim,)
        }

class PolicyNet(nn.Module):
    def __init__(self, embed_dim, num_patches, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(embed_dim * num_patches, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    def forward(self, z):
        return self.net(z)

def train():
    csv_path = '/home/aero/WM+SNCBF/Ref-policy/pendulum_dataset_controller.csv'
    embedding_dir = '/home/aero/WM+SNCBF/Ref-policy/dinov2_embeddings'
    batch_size = 64
    embed_dim = 768
    num_patches = 256
    action_dim = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter('runs/imitation_policy')
    # Prepare indices for train/val split
    full_dataset = PendulumImitationDataset(csv_path, embedding_dir)
    indices = np.arange(len(full_dataset))
    train_idx, val_idx = train_test_split(indices, test_size=0.1, random_state=42, shuffle=True)
    train_dataset = PendulumImitationDataset(csv_path, embedding_dir, indices=train_idx)
    val_dataset = PendulumImitationDataset(csv_path, embedding_dir, indices=val_idx)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    model = PolicyNet(embed_dim, num_patches, action_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    global_step = 0
    for epoch in range(20):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            z = batch['z'].to(device)
            action = batch['action'].to(device)
            pred_action = model(z)
            loss = F.mse_loss(pred_action, action)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar('Loss/train', loss.item(), global_step)
            global_step += 1
            pbar.set_postfix({'loss': loss.item()})
        print(f"Epoch {epoch}: train loss={loss.item():.4f}")
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                z = batch['z'].to(device)
                action = batch['action'].to(device)
                pred_action = model(z)
                val_loss = F.mse_loss(pred_action, action)
                val_losses.append(val_loss.item())
        val_loss_mean = np.mean(val_losses)
        writer.add_scalar('Loss/val', val_loss_mean, epoch)
        print(f"Epoch {epoch}: val loss={val_loss_mean:.4f}")
    torch.save(model.state_dict(), 'runs/imitation_policy/model_hiddim_128.pth')
    writer.close()

if __name__ == "__main__":
    train()
