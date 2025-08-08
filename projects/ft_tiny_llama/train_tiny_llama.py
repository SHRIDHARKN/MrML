import sys
sys.path.append("/home/voldemort/data_science/projects/dsml/MrML/projects")
from utils.params import CACHE_DIR
from utils.model_utils import load_pretrained_model, get_device, save_checkpoint
from utils.data_utils import generate_id, CSVDataset
from utils.params import LANGUAGE_MODELING_TASK
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from peft import LoraConfig, get_peft_model, PeftModel
import torch
from tqdm.auto import tqdm
import numpy as np

def prepare_prompt(text, label, tokenizer):
    prompt_template = f"Classify the emotion of the following text: {text}\n\nEmotion: {label}{tokenizer.eos_token}"
    return prompt_template
    
def collate_fn(batch, tokenizer, model_max_length):
    
    texts = [item['text'] for item in batch]
    labels = [item["label_name"] for item in batch]
    prompts = [prepare_prompt(text,label,tokenizer=tokenizer) for text,label in zip(texts,labels)]
    
    # Tokenize the entire batch of texts
    tokenized_inputs = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=model_max_length,
        return_tensors="pt"  # Returns PyTorch tensors
    )
    
    return tokenized_inputs


model_name = "TinyLlama/TinyLlama_v1.1"  # Replace with the actual model path
model_context_length = 512
batch_size=2
lr = 2e-3
num_epochs = 2
weight_decay = 0.001
debug = False
max_steps = 10
#is_peft_model = True
max_steps = max_steps if debug else np.inf
device = get_device()
tokenizer, model, model_tag = load_pretrained_model(model_name=model_name, 
                                                    cache_dir=CACHE_DIR, 
                                                    task=LANGUAGE_MODELING_TASK, 
                                                    device=device, 
                                                    )

model.gradient_checkpointing_enable()
#model = prepare_model_for_kbit_training(model)
model.config.use_cache = False
model.config.pretraining_tp = 1
model.resize_token_embeddings(len(tokenizer))

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)


model = get_peft_model(model, lora_config)


train_dataset = CSVDataset(root_dir="/mnt/g/dev/data/emotion/train")
val_dataset = CSVDataset(root_dir="/mnt/g/dev/data/emotion/val")

train_dataloader = DataLoader(
                            train_dataset, 
                            batch_size=batch_size, 
                            collate_fn=lambda examples: collate_fn(
                                examples, 
                                model_max_length=model_context_length, 
                                tokenizer=tokenizer)
                        )


#prepare_model_for_kbit_training()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

model.train()
progress_bar = tqdm(range(len(train_dataloader) * num_epochs))

checkpoint_save_epochs = 1
checkpoint_dir = f"/mnt/g/dev/model/ft_tiny_llama/{generate_id()}"
os.makedirs(checkpoint_dir, exist_ok=True)
is_peft_model = isinstance(model, PeftModel)
global_step = 0
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = batch.to(device)
        labels = batch["input_ids"].clone()
        outputs = model(**batch, labels=labels)
        loss = outputs.loss
        optimizer.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        global_step+=1
        if progress_bar.n % 10 == 0:
            print(f"Step {progress_bar.n}: Loss {loss.item()}")
        if global_step>max_steps:
            break
    # torch.save({
    #         'model_state_dict': model.state_dict()},checkpoint_dir)
            
    save_checkpoint(is_peft_model=is_peft_model,
                    epoch=epoch, checkpoint_save_epochs=1, 
                    checkpoint_dir=checkpoint_dir,
                    model=model, optimizer=optimizer, 
                    scheduler=None, avg_epoch_loss=loss,
                    global_step=None, model_name=None, lr=lr, 
                    weight_decay=weight_decay,
                    batch_size=batch_size,
                    num_epochs=num_epochs,
                    gradient_accumulation_steps=None,
                    model_context_length=model_context_length,label_context_length=None)
    print(f"Checkpoint saved @ {checkpoint_dir}")
    
print("Training finished!")
