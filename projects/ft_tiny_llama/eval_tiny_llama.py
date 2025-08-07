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
import re
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


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
    
    tokenized_inputs["actual_labels"] = labels
    return tokenized_inputs


model_name = "TinyLlama/TinyLlama_v1.1"  # Replace with the actual model path
model_context_length = 512
debug = True
max_steps = 10
batch_size = 4
max_steps = max_steps if debug else np.inf
device = get_device()
tokenizer, model, model_tag = load_pretrained_model(model_name=model_name, 
                                                    cache_dir=CACHE_DIR, 
                                                    task=LANGUAGE_MODELING_TASK, 
                                                    device=device, 
                                                    )

val_dataset = CSVDataset(root_dir="/mnt/g/dev/data/emotion/val")

val_dataloader = DataLoader(
                            val_dataset, 
                            batch_size=batch_size, 
                            collate_fn=lambda examples: collate_fn(
                                examples, 
                                model_max_length=model_context_length, 
                                tokenizer=tokenizer)
                        )

adapter_path = "/mnt/g/dev/model/ft_tiny_llama/gdobilerxxpnc5h90rha/adapter_epoch_2"
model = PeftModel.from_pretrained(model, adapter_path)
# This step is optional but recommended for deployment.
model = model.merge_and_unload() 

def preprocess_output_regex(regex, text):
    
    match = re.search(regex, text)
    emotion = "No match found"
    if match:
        emotion = match.group(1).strip()
        return emotion
    else:
       return emotion
   
res_ls = []
act_ls = []

loop = tqdm(val_dataloader, desc=">> Generating responses")
for batch in loop:
    outputs = model.generate(
                        input_ids=batch["input_ids"].to(model.device), 
                        max_new_tokens=10, 
                        do_sample=False
                    )
    response_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    response_text = [preprocess_output_regex(r"Emotion: (.*)",x) for x in response_text]

    res_ls.extend(response_text)
    act_ls.extend(batch["actual_labels"])
    
df_res = pd.DataFrame()
df_res["response"] = res_ls
df_res["actual"] = act_ls

print(df_res.head())

eval_fold = f"/mnt/g/dev/experiments/ft_tiny_llama/{generate_id()}"
os.makedirs(eval_fold, exist_ok=True)
eval_save_path = f"{eval_fold}/results.csv"

acc = accuracy_score(df_res["actual"], df_res["response"])
f1 = f1_score(df_res["actual"], df_res["response"], average="macro")
p = precision_score(df_res["actual"], df_res["response"], average="macro")
r = recall_score(df_res["actual"], df_res["response"], average="macro")
df_res.to_csv(eval_save_path, index=False)
print(f">> Eval results saved @ {eval_save_path}")
print(f"Accuracy: {acc}")
print(f"F1: {f1}")
print(f"Precision: {p}")
print(f"Recall: {r}")