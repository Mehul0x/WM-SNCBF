import torch
import torch.nn as nn

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