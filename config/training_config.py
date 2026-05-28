"""
Training configuration for NectarTRON fine-tuning.
Optimized for NVIDIA RTX Pro 2000 Blackwell with limited VRAM.
"""

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

AVAILABLE_MODELS = [
    "unsloth/Mistral-7B-bnb-4bit",  # Pre-quantized to fit in 8GB VRAM
    "unsloth/tinyllama-bnb-4bit",
]

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================

MAX_SEQ_LENGTH = 256  # Aggressive reduction for 8GB VRAM
BATCH_SIZE = 1  # Critical for limited VRAM
GRADIENT_ACCUMULATION_STEPS = 2  # Reduced to fit in 8GB
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4

# ============================================================================
# DATA & OUTPUT PATHS
# ============================================================================

DATASET_PATH = "./data/qa_dataset.jsonl"
OUTPUT_DIR = "./fine-tuned-model/nectartron-v0"

# ============================================================================
# TRAINING ARGUMENTS
# ============================================================================

TRAINING_ARGS = {
    "per_device_train_batch_size": BATCH_SIZE,
    "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
    "num_train_epochs": NUM_EPOCHS,
    "learning_rate": LEARNING_RATE,
    "fp16": False,
    "bf16": True,  # Better stability than fp16
    "optim": "adamw_8bit",  # Memory efficient optimizer
    "seed": 42,
    "logging_steps": 5,
    "save_steps": 50,
    "save_total_limit": 2,
    "report_to": [],  # No wandb/tensorboard
    "remove_unused_columns": False,
    "dataloader_pin_memory": False,
    "dataloader_num_workers": 0,
}

# ============================================================================
# INFERENCE TEST QUESTIONS
# ============================================================================

TEST_QUESTIONS = [
    "What trigger mode was used in run #635?",
    "Who conducted run #632?",
    "How many modules were used in run #621?",
]

# ============================================================================
# INFERENCE PARAMETERS
# ============================================================================

INFERENCE_PARAMS = {
    "max_length": 512,
    "top_p": 0.9,
    "temperature": 0.7,
    "do_sample": True,
}
