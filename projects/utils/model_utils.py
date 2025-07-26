from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from utils.params import LANGUAGE_MODELING_TASK
from utils.data_utils import log_msg
import torch
import warnings
warnings.simplefilter("ignore")


def load_img_2_text_hf_pipe(model_name = "Salesforce/blip-image-captioning-large", cache_dir=None):

    pipe = pipeline("image-to-text", model=model_name, model_kwargs={"cache_dir": cache_dir})
    log_msg(msg="Loading image-to-text pipeline")
    print(f"Model name: {model_name}")
    print(f"Cache directory: {cache_dir}")
    log_msg("Pipeline loaded successfully")
    return pipe
    

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
