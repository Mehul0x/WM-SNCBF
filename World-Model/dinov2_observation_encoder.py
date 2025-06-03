import os
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
from transformers import AutoImageProcessor, AutoModel

# Directory containing images
data_dir = 'World-Model/images'
output_dir = 'World-Model/dinov2_embeddings'
os.makedirs(output_dir, exist_ok=True)

# Load DINOv2 model and processor (ViT-large variant, can be changed)
model_name = 'facebook/dinov2-small'
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model.eval()
for param in model.parameters():
    param.requires_grad = False

def encode_image(image_path):
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs)
    # Patch embeddings: (batch, num_patches+1, embed_dim), remove CLS token
    patch_embeds = outputs.last_hidden_state[:, 1:, :].squeeze(0).cpu().numpy()
    return patch_embeds  # shape: (N, E)

if __name__ == "__main__":
    image_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.png')])
    for img_file in tqdm(image_files, desc="Encoding images with DINOv2"):
        img_path = os.path.join(data_dir, img_file)
        patch_embeds = encode_image(img_path)
        # Save as .npy file for each image
        np.save(os.path.join(output_dir, img_file.replace('.png', '.npy')), patch_embeds)
    print(f"Done. Embeddings saved to {output_dir}/")
