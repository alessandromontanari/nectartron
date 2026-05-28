import torch

from config.training_config import (
    AVAILABLE_MODELS,
    MAX_SEQ_LENGTH,
)

import logging
logger = logging.getLogger(__name__)

from utils.logging_config import log_and_print


def load_model_and_tokenizer():
    """Load pre-quantized model using Unsloth."""
    from unsloth import FastLanguageModel
    
    log_and_print(logger, "Loading model and tokenizer...")

    for ii, model in enumerate(AVAILABLE_MODELS):
        print(f"  - {ii}: {model}")
    print("We are looking into expanding the model options, considering the VRAM constraints...")
    model_name = input("Enter the index of the model to use: ").strip()
    if not model_name.isdigit() or int(model_name) >= len(AVAILABLE_MODELS):
        print("Invalid model index.")
        exit(1)
    model_name = AVAILABLE_MODELS[int(model_name)]

    # Load pre-quantized model (4-bit) for better VRAM efficiency
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,  # Better stability than float16
        load_in_4bit=True,    # 4-bit quantization
    )
    
    # Apply LoRA (Low-Rank Adaptation) for efficient fine-tuning
    log_and_print(logger, "Applying LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=8,  # Reduced rank for memory (was 16)
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", 
            "o_proj", "up_proj", "down_proj"
        ],
        bias="none",
        use_gradient_checkpointing=False,  # Disable to avoid Triton issues
        use_rslora=True,  # Stability with reduced rank
    )
    
    log_and_print(logger, "✓ Model loaded and LoRA applied")
    return model, tokenizer, model_name