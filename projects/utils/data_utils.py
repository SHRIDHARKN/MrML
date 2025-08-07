import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


import os
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

import random
import string
from termcolor import colored

import pandas as pd
import numpy as np

def show_blue_msg(msg):
    print(f"[blue]{msg}[/blue]")

def show_reg_msg(msg):
    print(f"[red]{msg}[/red]")

def show_green_msg(msg):
    print(f"[green]{msg}[/green]")

def show_orange_msg(msg):
    print(f"[orange]{msg}[/orange]")

def show_pink_msg(msg):
    print(f"[pink]{msg}[/pink]")

def get_img_dataloader(data_dir, img_height, img_width, batch_size, data_tag="data", help=False):

    if help:
        return """
        train_loader, test_loader, val_loader = get_train_test_val_img_dataloaders(data_dir, img_height, img_width,batch_size)
        """
    show_blue_msg(f"data_dir path: {data_dir}")
    show_blue_msg(f"found subfolds: {os.listdir(data_dir)}")
    show_reg_msg(f"img_height, img_weight: {img_height}, {img_width}")
    show_reg_msg(f"batch_size: {batch_size}")
        
    data_transforms = {
        data_tag: transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor()
        ]),
    }

    image_datasets = {data_tag: datasets.ImageFolder(data_dir, data_transforms[data_tag])}
    show_blue_msg(f"image loader is ready")

    dataloaders = {data_tag: DataLoader(image_datasets[data_tag], batch_size=batch_size, shuffle=True, num_workers=4)}

    img_loader = dataloaders[data_tag]
    show_blue_msg(f"img_loader is ready")

    return img_loader


def get_train_test_val_img_dataloaders(data_dir, img_height, img_width, batch_size, help=False):

    if help:
        return """
        train_loader, test_loader, val_loader = get_train_test_val_img_dataloaders(data_dir, img_height, img_width,batch_size)
        """
    show_blue_msg(f"data_dir path: {data_dir}")
    show_blue_msg(f"found subfolds: {os.listdir(data_dir)}")
    show_reg_msg(f"img_height, img_weight: {img_height}, {img_width}")
    show_reg_msg(f"batch_size: {batch_size}")
        
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor()
        ]),
        'val': transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor()
        ]),
        'test': transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor()
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val', 'test']}
    show_blue_msg(f"image loader is ready")
    
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val', 'test']}

    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    test_loader = dataloaders['test']
    show_blue_msg(f"train_loader is ready")
    show_blue_msg(f"val_loader is ready")
    show_blue_msg(f"test_loader is ready")
    
    return train_loader, test_loader, val_loader


class SingleClassImageDataset(Dataset):
    def __init__(self, data_dir, img_height, img_width, debug=False):
        self.data_dir = data_dir
        self.img_paths = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir)
                          if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if debug:
            self.img_paths = self.img_paths[:10]

        self.transform = transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img


class ImageTextEmbeddingDataset(Dataset):
    def __init__(self, csv_file: str, transform=None):
        
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform
        # self.data_frame['text_embed'] = self.data_frame['text_embed'].apply(parse_embed)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.data_frame.iloc[idx]
        img_full_path = row['full_paths']
        image = Image.open(img_full_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        description_text = row['text']
        text_embedding = self.parse_embed(row['text_embed'])
        text_embedding = torch.tensor(text_embedding, dtype=torch.float32)
        return {
            'image': image,
            'text_embedding': text_embedding,
            'description_text': description_text,
            'img_full_path': img_full_path
        }

    def parse_embed(self,embed_str):
        embed_clean = embed_str.strip("[]")
        return np.fromstring(embed_clean, sep=' ')
    

def get_img_dataloader(data_dir, img_height, img_width, batch_size, debug=False, help=False):
    if help:
        return """img_loader = get_img_dataloader(data_dir, img_height, img_width, batch_size)"""

    print(f"📁 data_dir path: {data_dir}")
    print(f"🖼️ found files: {len(os.listdir(data_dir))}")
    print(f"📐 img_height, img_width: {img_height}, {img_width}")
    print(f"📦 batch_size: {batch_size}")

    dataset = SingleClassImageDataset(data_dir=data_dir, img_height=img_height, img_width=img_width, debug=debug)
    img_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    print(f"✅ img_loader is ready")
    return img_loader


def generate_id(length=20):
    """Generate a random alphanumeric ID of specified length."""
    alpha_num = string.ascii_letters+string.digits
    _id = "".join(random.choice(alpha_num) for _ in range(length))
    return _id.lower()


# def log_msg(msg, msg_typ="normal", font_color="white", bg_color="green", header="", sep = "="):
    
#     # msg = " \u039B | " +" "*2+ msg +" "*10
    
#     if msg_typ == "progress":
#         print(colored(sep * 10 + " INFO " + sep * 10, 'black', 'on_white', attrs=['bold'])) 
#         print(colored(msg, 'black', 'on_white', attrs=['bold']))   
#     elif msg_typ == "data":
#         print(colored(sep * 10 + " DATA/ VARS / CONFIG " + sep * 10, 'black', 'on_white', attrs=['bold'])) 
#         print(colored(msg, 'white', 'on_blue', attrs=['bold']))
#     elif msg_typ == "normal":
#         print(colored(sep * 10 + " PROGRESS/ SAVES " + sep * 10, 'black', 'on_white', attrs=['bold'])) 
#         print(colored(msg, 'white', 'on_green', attrs=['bold']))
#     else:
#         print(colored(sep * 10 + header + sep * 10, 'black', 'on_white', attrs=['bold'])) 
#         print(colored(msg, font_color, f'on_{bg_color}', attrs=['bold']))
#     print(colored(sep * 50+"\n", 'black', 'on_white', attrs=['bold'])) 

def log_msg(msg, font_color="black", bg_color="white"):
    print(colored(msg, font_color, f'on_{bg_color}', attrs=['bold']))
    
class CSVDataset(Dataset):
    def __init__(self, root_dir):
        
        self.root_dir = root_dir
        self.csv_files = [f for f in os.listdir(root_dir) if f.endswith('.csv')][:1]
        self.all_data = []
        self._load_metadata()
        self.num_files = len(self.csv_files)
        print(f"Found {self.num_files} CSV files in {self.root_dir}")

    def _load_metadata(self):
        
        for file_idx, filename in enumerate(self.csv_files):
            filepath = os.path.join(self.root_dir, filename)
            df = pd.read_csv(filepath, header=0)  # Assuming your CSV has a header row
            for row_idx in range(len(df)):
                self.all_data.append((file_idx, row_idx))
                
    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        file_idx, row_idx = self.all_data[idx]
        filename = self.csv_files[file_idx]
        filepath = os.path.join(self.root_dir, filename)
        df = pd.read_csv(filepath, header=0)
        sample = df.iloc[row_idx].to_dict()  # Get the row as a dictionary
        return sample
