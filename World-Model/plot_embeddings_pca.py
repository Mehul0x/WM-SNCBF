import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Settings
csv_path = 'pendulum_dataset.csv'
embedding_dir = 'dinov2_embeddings'

# Load CSV
csv = pd.read_csv(csv_path)

# Filter by class
safe = csv[csv['category'] == 'safe']
unsafe = csv[csv['category'] == 'unsafe']
buffer = csv[csv['category'] == 'buffer']

# Sample
# safe_sample = safe.sample(n=min(500, len(safe)), random_state=42)
safe_sample=safe
unsafe_sample = unsafe.sample(n=min(700, len(unsafe)), random_state=42)
# unsafe_sample=unsafe
# rest_needed =  len(safe_sample) + len(unsafe_sample)
buffer_sample = buffer.sample(n=min(700, len(buffer)), random_state=42)

# Combine
df = pd.concat([safe_sample, unsafe_sample, buffer_sample]).reset_index(drop=True)

# Load embeddings (flatten all patch tokens)
embeddings = []
labels = []
for _, row in df.iterrows():
    emb_path = os.path.join(embedding_dir, row['state_image'].replace('.png', '.npy'))
    emb = np.load(emb_path)
    emb_flat = emb.flatten()
    embeddings.append(emb_flat)
    labels.append(row['category'])
embeddings = np.stack(embeddings)

print(emb.shape)
# PCA
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

# Plot
plt.figure(figsize=(8, 6))
colors = {'safe': 'green', 'unsafe': 'red', 'buffer': 'black'}
for cat in ['safe', 'unsafe', 'buffer']:
    idx = [i for i, l in enumerate(labels) if l == cat]
    plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], label=cat, alpha=0.7, c=colors[cat])
plt.legend()
plt.title(f'PCA of DINOv2 Patch Embeddings ({len(safe_sample)+len(unsafe_sample)+len(buffer_sample)} samples)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.tight_layout()
plt.show()
