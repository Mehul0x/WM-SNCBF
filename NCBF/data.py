import os
import pandas as pd
import numpy as np
import torch

def gen_batch_data():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    csv_path = os.path.join(os.path.dirname(__file__), '../World-Model/pendulum_dataset.csv')
    embedding_dir = os.path.join(os.path.dirname(__file__), '../World-Model/dinov2_embeddings')
    df = pd.read_csv(csv_path)
    
    def get_embedding(state_img):
        npy_name = state_img.replace('.png', '.npy')
        npy_path = os.path.join(embedding_dir, npy_name)
        return torch.tensor(np.load(npy_path), dtype=torch.float64).to(device)

    init = []
    unsafe = []
    domain = []
    for _, row in df.iterrows():
        embedding = get_embedding(row['state_image'])
        if row['category'] == 'safe':
            init.append(embedding)
        if row['category'] == 'unsafe':
            unsafe.append(embedding)
        domain.append(embedding)

    return init, unsafe, domain
