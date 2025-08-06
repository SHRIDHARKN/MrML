from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from utils.params import LANGUAGE_MODELING_TASK
from transformers import BitsAndBytesConfig
from utils.data_utils import log_msg
import torch
import os
import warnings
warnings.simplefilter("ignore")

bnb_config_4bit = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type = 'nf4',
    bnb_4bit_compute_dtype = torch.bfloat16
)

bnb_config_8bit = BitsAndBytesConfig(
    load_in_8bit = True
)

def load_img_2_text_hf_pipe(model_name = "Salesforce/blip-image-captioning-large", cache_dir=None):

    pipe = pipeline("image-to-text", model=model_name, model_kwargs={"cache_dir": cache_dir})
    log_msg(msg="Loading image-to-text pipeline")
    log_msg(f"Model name: {model_name}")
    log_msg(f"Cache directory: {cache_dir}")
    log_msg("Pipeline loaded successfully")
    return pipe
    

def save_df_in_parts(df, records_per_file, data_tag):

    for idx,i in enumerate(range(0,len(df),records_per_file)):
        df_sample = df.iloc[i:i+records_per_file]
        save_path = f"/mnt/g/dev/data/emotion/{data_tag}/data_part_{idx+1}.csv"
        df_sample.to_csv(save_path)
        print(f">> File data_part_{idx+1}.csv save success.")

# def load_pretrained_model(model_name, cache_dir, task):
#     """
#     Load a pretrained model and tokenizer from Hugging Face.
    
#     Args:
#         model_name (str): The name of the model to load.
        
#     Returns:
#         tuple: A tuple containing the tokenizer and model.
#     """
#     print("=== Loading pretrained model and tokenizer ===")
#     print(f"Model name: {model_name}")
#     print(f"Cache directory: {cache_dir}")
#     print(f"Task: {task}")
    
#     if task==LANGUAGE_MODELING_TASK:
#         tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
#         model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
        
#         tokenizer.pad_token = tokenizer.eos_token
#         model.config.pad_token_id = tokenizer.pad_token_id

#         model_tag = model_name.split("/")[-1] # Extract model name for logging
#         print(f"Model tag: {model_tag}")
#         print("=== Model and tokenizer loaded successfully ===")
#         return tokenizer, model, model_tag

def save_checkpoint(epoch, checkpoint_dir, avg_epoch_loss,
                    model, optimizer=None, scheduler=None, 
                    global_step=None, model_name=None, lr=None, 
                    weight_decay=None, batch_size=None, num_epochs=None,
                    gradient_accumulation_steps=None,
                    model_context_length=None,label_context_length=None,
                    checkpoint_save_epochs=1):
    
    if (epoch + 1) % checkpoint_save_epochs == 0:
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
        # Save both model state_dict and optimizer/scheduler states for full resume capability
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'loss': avg_epoch_loss,
            'global_step': global_step if global_step else None,
            'model_name': model_name if model_name else None,
            'lr': lr if lr else None,
            'weight_decay': weight_decay if weight_decay else None,
            'gradient_accumulation_steps': gradient_accumulation_steps if gradient_accumulation_steps else None,
            'batch_size': batch_size if batch_size else None,
            'num_epochs': num_epochs if num_epochs else None,
            'model_context_length': model_context_length if model_context_length else None,
            'label_context_length': label_context_length if label_context_length else None,
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")


def get_device():
    """Returns the device to be used for training."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_pretrained_model(model_name, cache_dir, task, device, bnb_config=None, quant=""):

    """
    Load a pretrained model and tokenizer from Hugging Face.
    
    Args:
        model_name (str): The name of the model to load.
        cache_dir (str): The directory to cache the model.
        task (str): The task for which the model is being loaded.
        bnb_config (BitsAndBytesConfig, optional): Configuration for 4-bit quantization.

    Returns:
        tuple: A tuple containing the tokenizer and model.
    """
    log_msg("Loading pretrained model and tokenizer")
    log_msg(f"Model name: {model_name}")
    log_msg(f"Cache directory: {cache_dir}")
    log_msg(f"Task: {task}")

    if bnb_config:
        quant_4bit = bnb_config._load_in_4bit
        quant_8bit = bnb_config._load_in_8bit
        quant = "_4bit_quant" if quant_4bit else "_8bit_quant" if quant_8bit else ""
        log_msg(msg=f"Quantization config: {quant}")

    if task==LANGUAGE_MODELING_TASK:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                     cache_dir=cache_dir, 
                                                     quantization_config=bnb_config,
                                                     device_map="auto" if bnb_config else device)

        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

        model_tag = model_name.split("/")[-1] # Extract model name for logging
        model_tag = model_tag+quant if bnb_config else model_tag
        log_msg(f"Model tag: {model_tag}")
        log_msg("Model and tokenizer loaded successfully")
        return tokenizer, model, model_tag
