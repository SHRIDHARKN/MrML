from pdb import run
from utils.params import CACHE_DIR, EXPERIMENT_CACHE_DIR
from utils.model_utils import load_img_2_text_hf_pipe, get_device
import os
import pandas as pd
from utils.data_utils import log_msg, generate_id
import click

def generate_descriptions(model_name, img_folder, output_dir=None, cache_dir=CACHE_DIR, debug=False):
    """
    Main function to load the image-to-text pipeline.
    
    Args:
        model_name (str): The name of the model to load.
        img_folder (str): The folder containing images to process.
        cache_dir (str): The directory to cache the model.
        debug (bool): If True, prints debug information.
    """
    num_imgs = len(os.listdir(img_folder))
    log_msg(f"Processing {num_imgs} images from {img_folder} using model {model_name}...")
    device = get_device()
    
    pipe = load_img_2_text_hf_pipe(model_name=model_name, cache_dir=cache_dir)
    
    full_paths = [f"{img_folder}/{img}" for img in os.listdir(img_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]
    desc = pipe(full_paths)
    
    if output_dir is None:
        output_dir = os.path.join(EXPERIMENT_CACHE_DIR, generate_id())
    
    if debug:
        log_msg(f"Image folder: {img_folder}")
        log_msg(f"Full paths: {full_paths}")
        log_msg(f"Descriptions: {desc}")
    else:
        os.makedirs(output_dir, exist_ok=True)
        df = pd.DataFrame({"Image": full_paths, "Description": desc})
        df.to_csv(f"{output_dir}/image_descriptions.csv", index=False)
        
    log_msg(f"Descriptions saved to {output_dir}/image_descriptions.csv")
    
@click.command()
@click.option('--model_name', default="Salesforce/blip-image-captioning-large", help='Name of the model to load.')
@click.option('--img_folder', required=True, help='Folder containing images to process.')
@click.option('--output_dir', default=None, help='Directory to save the output CSV file.')
@click.option('--cache_dir', default=CACHE_DIR, help='Directory to cache the model.')
@click.option('--debug', default=True, help='Enable debug mode for additional output.')
def main(model_name, img_folder, output_dir, cache_dir, debug):
    """
    Command-line interface to run the image-to-text pipeline.
    
    Args:
        model_name (str): The name of the model to load.
        img_folder (str): The folder containing images to process.
        output_dir (str): Directory to save the output CSV file.
        cache_dir (str): Directory to cache the model.
        debug (bool): If True, prints debug information.
    """
    generate_descriptions(model_name, img_folder, output_dir, cache_dir, debug)

if __name__ == "__main__":
    main()
