import pandas as pandas
import numpy as np
import os
from torch.utils.data import Dataset
import torch

class dataset(Dataset):
    def __init__(self, csv_file, embedding_dir):
        self.df = pandas.read_csv(csv_file)
        self.embedding_dir = embedding_dir
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        emb = self.get_embedding(row['state_image'])
        label = row['category']

        return emb, label
    
    def __len__(self):
        return self.df.shape[0]
    
    def get_embedding(self, state_img):
        npy_name = state_img.replace('.png', '.npy')
        npy_path = os.path.join(self.embedding_dir, npy_name)
        return torch.tensor(np.load(npy_path), dtype=torch.float64).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))