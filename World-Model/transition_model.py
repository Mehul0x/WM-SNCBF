import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import List
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# --- Data Utilities ---
class PendulumLatentDataset(Dataset):
    def __init__(self, csv_path, embedding_dir, context_len=8):
        self.df = pd.read_csv(csv_path)
        self.embedding_dir = embedding_dir
        self.context_len = context_len
        self.trajs = self._split_trajectories()
        self.samples = self._make_samples()

    def _split_trajectories(self):
        # Split into continuous chunks of 200
        trajs = []
        for start in range(0, len(self.df), 200):
            trajs.append(self.df.iloc[start:start+200].reset_index(drop=True))
        return trajs

    def _make_samples(self):
        samples = []
        for traj in self.trajs:
                for i in range(self.context_len, len(traj)-1):
                    # Each sample: (z_{t-H:t-1}, a_{t-H:t-1}, z_t, a_t, z_{t+1})
                    samples.append((traj, i))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        traj, i = self.samples[idx]
        # Get context
        zs = []
        actions = []
        for j in range(i-self.context_len, i):
            z = np.load(os.path.join(self.embedding_dir, traj.loc[j, 'state_image'].replace('.png', '.npy')))
            zs.append(z)
            a = np.array(eval(traj.loc[j, 'action']), dtype=np.float32)
            actions.append(a)
        # Current z_t, a_t, next z_{t+1}
        z_t = np.load(os.path.join(self.embedding_dir, traj.loc[i, 'state_image'].replace('.png', '.npy')))
        a_t = np.array(eval(traj.loc[i, 'action']), dtype=np.float32)
        z_next = np.load(os.path.join(self.embedding_dir, traj.loc[i+1, 'state_image'].replace('.png', '.npy')))
        # Optionally, proprioception: obs = eval(traj.loc[i, 'observation'])
        return {
            'z_context': torch.tensor(np.stack(zs), dtype=torch.float32), # (H, N, E)
            'a_context': torch.tensor(np.stack(actions), dtype=torch.float32), # (H, action_dim)
            'z_t': torch.tensor(z_t, dtype=torch.float32), # (N, E)
            'a_t': torch.tensor(a_t, dtype=torch.float32), # (action_dim,)
            'z_next': torch.tensor(z_next, dtype=torch.float32), # (N, E)
        }

# --- Model ---
class ActionMLP(nn.Module):
    def __init__(self, action_dim, embed_dim, hidden_dim=128): #the action dim in pendulum is 1, is a hidden dim of 128 appropriate?
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(), #is this the right activation?
            nn.Linear(hidden_dim, embed_dim)
        )
    def forward(self, a):
        return self.net(a)

class TransitionViT(nn.Module):
    def __init__(self, embed_dim, num_patches, action_dim, context_len, nhead=8, num_layers=4, mlp_dim=1024):
        super().__init__()
        self.context_len = context_len
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.action_encoder = ActionMLP(action_dim, embed_dim)
        # Positional encoding for time and patch
        self.time_pos = nn.Parameter(torch.randn(context_len+1, 1, embed_dim))
        self.patch_pos = nn.Parameter(torch.randn(1, num_patches, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=mlp_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, z_context, a_context, z_t, a_t):
        # z_context: (B, H, N, E), a_context: (B, H, action_dim), z_t: (B, N, E), a_t: (B, action_dim)
        B, H, N, E = z_context.shape
        # Encode actions
        a_context_enc = self.action_encoder(a_context.view(B*H, -1)).view(B, H, 1, E).expand(-1, -1, N, -1)
        #(B*H,E) -> (B,H,1,E) -> (B,H,N,E)
        
        z_context = z_context + a_context_enc # (B, H, N, E)
        # Current step
        a_t_enc = self.action_encoder(a_t).unsqueeze(1).expand(-1, N, -1)
        #(B,E) -> (B,1,E) -> (B,N,E)
        z_t = z_t + a_t_enc # (B, N, E)
        # Stack all: (B, H+1, N, E)
        z_seq = torch.cat([z_context, z_t.unsqueeze(1)], dim=1)
        # Add time and patch pos
        z_seq = z_seq + self.time_pos[:H+1] + self.patch_pos
        # Merge time and patch: (B, (H+1)*N, E)
        z_seq = z_seq.view(B, (H+1)*N, E)
        # Causal mask: only attend to previous time steps
        seq_len = (H+1)*N
        mask = torch.triu(torch.ones(seq_len, seq_len, device=z_seq.device), diagonal=1).bool()
        # print("kuch toh ho raha")

        z_out = self.transformer(z_seq, mask=mask) 
        # Take last frame's patch tokens: (B, N, E)

        z_pred = z_out[:, -N:, :]
        z_pred = self.out_proj(z_pred)
        return z_pred

# --- Training Loop (sketch) ---
def train():
    # Hyperparameters
    context_len = 8
    batch_size = 8
    embed_dim = 768  # DINOv2-base
    num_patches = 256  # 7x7 for 224x224 images
    action_dim = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter('runs/transition_model')
    dataset = PendulumLatentDataset('pendulum_dataset.csv', 'dinov2_embeddings', context_len=context_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = TransitionViT(embed_dim, num_patches, action_dim, context_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    global_step = 0
    for epoch in range(10):
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            z_context = batch['z_context'].to(device) # (B, H, N, E)
            a_context = batch['a_context'].to(device) # (B, H, action_dim)
            z_t = batch['z_t'].to(device) # (B, N, E)
            a_t = batch['a_t'].to(device) # (B, action_dim)
            z_next = batch['z_next'].to(device) # (B, N, E)
            z_pred = model(z_context, a_context, z_t, a_t)
            loss = F.mse_loss(z_pred, z_next)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar('Loss/train', loss.item(), global_step)
            global_step += 1
            pbar.set_postfix({'loss': loss.item()})
        print(f"Epoch {epoch}: loss={loss.item():.4f}")
    # Save the final model after all epochs
    torch.save(model.state_dict(), 'runs/transition_model/model_final.pth')
    writer.close()

if __name__ == "__main__":
    train()