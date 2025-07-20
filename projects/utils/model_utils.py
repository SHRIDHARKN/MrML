from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from utils.params import LANGUAGE_MODELING_TASK
import warnings
warnings.simplefilter("ignore")


def load_img_2_text_hf_pipe(model_name = "Salesforce/blip-image-captioning-large", cache_dir=None):

    pipe = pipeline("image-to-text", model=model_name, model_kwargs={"cache_dir": cache_dir})
    print("=== Loading image-to-text pipeline ===")
    print(f"Model name: {model_name}")
    print(f"Cache directory: {cache_dir}")
    print("=== Pipeline loaded successfully ===")
    return pipe
    

def load_pretrained_model(model_name, cache_dir, task):
    """
    Load a pretrained model and tokenizer from Hugging Face.
    
    Args:
        model_name (str): The name of the model to load.
        
    Returns:
        tuple: A tuple containing the tokenizer and model.
    """
    print("=== Loading pretrained model and tokenizer ===")
    print(f"Model name: {model_name}")
    print(f"Cache directory: {cache_dir}")
    print(f"Task: {task}")
    
    if task==LANGUAGE_MODELING_TASK:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
        
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

        model_tag = model_name.split("/")[-1] # Extract model name for logging
        print(f"Model tag: {model_tag}")
        print("=== Model and tokenizer loaded successfully ===")
        return tokenizer, model, model_tag
