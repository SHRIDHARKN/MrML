import os
os.chdir("..")
from utils.model_utils import load_pretrained_model
from utils.params import CACHE_DIR, LANGUAGE_MODELING_TASK
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from trl.core import respond_to_batch # Helper to generate text from the model
import torch


model_name = "gpt2"

tokenizer, model, model_tag = load_pretrained_model(model_name=model_name,
                                                    cache_dir=CACHE_DIR,
                                                    task=LANGUAGE_MODELING_TASK)

def evaluate_syllogism_conclusion(premises, generated_conclusion, ground_truth_conclusion):
    # This is a placeholder. You need to implement actual syllogistic logic here.
    # For a small dataset, you might map prompts to expected correct answers.

    # Simplified example: direct string comparison (very brittle for real reasoning)
    if generated_conclusion.strip().lower() == ground_truth_conclusion.strip().lower():
        return 1.0  # Correct
    else:
        # Penalize for incorrect conclusions or nonsense
        # You could add negative rewards for incorrect validity, or for generating something when
        # it should be "no conclusion", etc.
        return -1.0 # Incorrect or invalid
    

# 1. Prepare your dataset
# Example:
syllogism_data = [
    {"prompt": "Premise 1: All men are mortal. Premise 2: Socrates is a man. Conclusion:", "ground_truth": "Socrates is mortal.", "valid": True},
    {"prompt": "Premise 1: All birds fly. Premise 2: A penguin is a bird. Conclusion:", "ground_truth": "No conclusion.", "valid": False}, # Example of invalid conclusion from premises
    # Add more examples for your small dataset
]

# Convert to a format suitable for PPOTrainer queries
# For each example, you need the query (prompt) that the model will complete
queries = [d["prompt"] for d in syllogism_data]
# Store ground truths and validity for reward calculation
ground_truths = [d["ground_truth"] for d in syllogism_data]
validities = [d["valid"] for d in syllogism_data]


# 2. Load your model (and value head model)
model_with_value_head = AutoModelForCausalLMWithValueHead.from_pretrained(model_name, cache_dir=CACHE_DIR)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name, cache_dir=CACHE_DIR)

model_with_value_head.config.pad_token_id = tokenizer.pad_token_id
ref_model.config.pad_token_id = tokenizer.pad_token_id


# 3. Configure PPO Trainer
ppo_config = PPOConfig(
    learning_rate=1e-5,
    ppo_epochs=4, # Number of epochs to train the PPO policy
    mini_batch_size=1, # Very small for small dataset to see immediate effects
    batch_size=1,    # Must be >= mini_batch_size
    gradient_accumulation_steps=1,
    target_kl=0.1, # KL divergence constraint
    init_kl_coef=0.01,
    seed=42,
    # You might need to adjust these hyperparams significantly for reasoning
)


ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model_with_value_head,
    ref_model=ref_model, # Only needed for PPO, not DPO or GRPO (if GRPO doesn't use it)
    tokenizer=tokenizer,
)