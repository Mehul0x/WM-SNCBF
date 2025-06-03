import pandas as pandas
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader

class dataset(Dataset):
    def __init__(self, csv_file, embedding_dir):
        self.df = pandas.read_csv(csv_file)
        self.embedding_dir = embedding_dir
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        emb = self.get_embedding(row['state_img'])
        label = row['label']

        return emb, label
    
    def __len__(self):
        return self.df.shape[0]
    
    def get_embedding(self, state_img):
        npy_name = state_img.replace('.png', '.npy')
        npy_path = os.path.join(self.embedding_dir, npy_name)
        return np.load(npy_path)
    
dataloader=DataLoader(dataset=dataset(csv_file='data.csv', embedding_dir='embeddings'),
                     batch_size=32, shuffle=True, num_workers=4)