import torch
from pathlib import Path
from typing import List, Dict
from config.training_config import (
    MAX_SEQ_LENGTH,
    TRAINING_ARGS,
    INFERENCE_PARAMS
)

import logging
logger = logging.getLogger(__name__)


from utils.logging_config import log_and_print


def tokenize_dataset(texts: List[str], tokenizer, max_length: int) -> Dict:
    """Tokenize texts for training."""
    log_and_print(logger, "Tokenizing dataset...")
    
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    
    # Create labels: shift input IDs for causal language modeling
    labels = tokenized["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100  # Ignore padding
    labels = labels.roll(shifts=-1, dims=1)
    labels[:, -1] = -100  # Ignore last token
    
    tokenized["labels"] = labels
    log_and_print(logger, f"✓ Tokenized {len(texts)} samples")
    
    return tokenized


def train_model(model, tokenizer, formatted_texts: List[str], output_dir: str):
    """Fine-tune the model on the dataset."""
    from transformers import (
        TrainingArguments, 
        Trainer, 
        DataCollatorForLanguageModeling
    )
    from datasets import Dataset
    
    # Tokenize
    tokenized = tokenize_dataset(formatted_texts, tokenizer, MAX_SEQ_LENGTH)
    dataset = Dataset.from_dict({
        "input_ids": tokenized["input_ids"],
        "labels": tokenized["labels"],
    })
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        **TRAINING_ARGS,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    logger.info("Starting fine-tuning...")
    
    trainer.train()
    
    log_and_print(logger, "✓ Training completed!")
    
    return model, tokenizer


def save_model(model, tokenizer, output_dir: str):
    """Save fine-tuned model and tokenizer."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    log_and_print(logger, f"Saving model to {output_dir}...")
    # Save LoRA adapters
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    log_and_print(logger, f"✓ Model saved to {output_dir}!")


def test_inference(model, tokenizer, test_questions: List[str]):
    """Test the fine-tuned model on sample questions."""
    log_and_print(logger, "Testing inference on sample questions...")
    log_and_print(logger, "="*60 + "\n")
    
    for question in test_questions:
        prompt = f"<s>[INST] {question} [/INST]"
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **INFERENCE_PARAMS,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(prompt, "").strip()
        
        log_and_print(logger, f"Q: {question}")
        log_and_print(logger, f"A: {response}\n")