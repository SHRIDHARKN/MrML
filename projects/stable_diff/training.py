import sys
sys.path.append("/home/voldemort/data_science/projects/dsml/MrML/projects")
from stable_diff_v1 import DiffusionModel
from torchvision import transforms
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from utils.data_utils import generate_id
from utils.data_utils import ImageTextEmbeddingDataset
import os
torch.cuda.empty_cache()

def get_img_pairs(imgs,noise_levels):
    
    img_a_ls = []
    img_b_ls = []
    time_ls = []
    #noise = torch.rand_like(imgs[0])
    noise = torch.rand_like(imgs[0])

    for img in imgs:
        for i in range(len(noise_levels)-1):
            img_a = img + noise*noise_levels[i]
            img_b = img + noise*noise_levels[i+1]
            img_a_ls.append(img_a)
            img_b_ls.append(img_b)
            time_ls.append(float(i))
    
    return torch.stack(img_a_ls), torch.stack(img_b_ls), torch.tensor(time_ls) 

def start_training(image_text_loader,diffusion_model,optimizer, device ,loss_fn, n_epochs=500, image_key="image", text_key="text_embedding"):
    
    for epoch in range(n_epochs):
        for batch_idx, batch in enumerate(image_text_loader):
            images = batch[image_key]
            text_embeddings = batch[text_key]
            images = images.to(device)
            text_embeddings = text_embeddings.repeat_interleave(4, dim=0)  # shape: (24, 384)
            text_embeddings = text_embeddings.to(device)
            
            img_a, img_b, time_ls = get_img_pairs(imgs=images,noise_levels=noise_levels)
            img_a = img_a.to(device=device)
            img_b = img_b.to(device=device)
            time_ls = time_ls.unsqueeze(-1).to(device=device)
            
            optimizer.zero_grad()

            preds = diffusion_model(img_inp=img_a, time_info=time_ls, text_inp=text_embeddings)

            loss = loss_fn(preds, img_b)

            loss.backward()
            optimizer.step()
            print(f" Epoch {epoch+1} Batch {batch_idx+1}/{len(image_text_loader)}: Loss = {loss.item():.4f}")

    return diffusion_model, optimizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

img_size = 128
img_channels = 3
text_embed_dim=384
time_embed_dim=img_size
batch_size = 4
lr=1e-4
weight_decay=0.001
timesteps = 4
noise_levels = torch.linspace(1,0,timesteps+1)
n_epochs=100

data_path = "/mnt/g/dev/data/alea/kqk08n2j47clcyyky3jq/txt_embed/image_descriptions.csv"

image_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)), # Resize images to your desired input size
    transforms.ToTensor(),         # Converts PIL Image to Tensor (H, W, C) -> (C, H, W), normalizes to [0, 1]
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalize to [-1, 1]
])


# --- 4. Instantiate Dataset and DataLoader ---
dataset = ImageTextEmbeddingDataset(
    csv_file=data_path,
    transform=image_transform
)

image_text_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
diffusion_model = DiffusionModel(img_channels=img_channels,
                                 img_size=img_size,
                                 text_embed_dim=text_embed_dim,
                                 time_embed_dim=time_embed_dim
                                 )

diffusion_model.to(device)
# Optimizer and Loss
optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=lr, weight_decay=weight_decay)
loss_fn = nn.MSELoss()

# train model

diffusion_model, optimizer = start_training(image_text_loader=image_text_loader,
                                 diffusion_model=diffusion_model,
                                 optimizer=optimizer, 
                                 device=device ,
                                 loss_fn=loss_fn, 
                                 n_epochs=n_epochs, 
                                 image_key="image", 
                                 text_key="text_embedding"
                                )

# save model
exp_id = generate_id()
diffusion_model_path = f"/mnt/g/dev/model/diffusion_model/{exp_id}/model/alea_diffusion_model.pth"
diffusion_opt_path = f"/mnt/g/dev/model/diffusion_model/{exp_id}/opt/alea_diffusion_opt.pth"
os.makedirs(os.path.dirname(diffusion_model_path), exist_ok=True)
os.makedirs(os.path.dirname(diffusion_opt_path), exist_ok=True)
print(f"Saving model to {diffusion_model_path}...")
torch.save(diffusion_model.state_dict(), diffusion_model_path)
print(f"Saving opt to {diffusion_opt_path}...")
torch.save(optimizer.state_dict(), diffusion_opt_path)
print("Traniing complete.")