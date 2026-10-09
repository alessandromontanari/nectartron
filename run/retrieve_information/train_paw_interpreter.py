import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from utils.dataset import load_qa_dataset, format_qa_for_training


# TODO: add logging

def load_training_config():

    MODEL_NAME = "Qwen/Qwen3-0.6B"  # ~375MB in 4-bit
    OUTPUT_DIR = "./fine-tuned-interpreter-8gb-unsloth"
    MAX_SEQ_LENGTH = 256            # Critical for 8GB
    BATCH_SIZE = 1                 # Must be 1 for 8GB
    GRAD_ACCUM_STEPS = 4           # Simulate batch_size=4
    EPOCHS = 2
    LEARNING_RATE = 2e-3
    USE_4BIT = True               # Enable 4-bit quantization
    USE_BF16 = True               # Better precision than fp16

    return {
        "model_name": MODEL_NAME,
        "output_dir": OUTPUT_DIR,
        "max_seq_length": MAX_SEQ_LENGTH,
        "batch_size": BATCH_SIZE,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "use_4bit": USE_4BIT,
        "use_bf16": USE_BF16,
    }



def load_model_and_tokenizer():
    """Load pre-quantized model using Unsloth."""

    config = load_training_config()
    MODEL_NAME = config["model_name"]
    MAX_SEQ_LENGTH = config["max_seq_length"]
    USE_4BIT = config["use_4bit"]
    USE_BF16 = config["use_bf16"]

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16 if USE_BF16 else None,
        load_in_4bit=USE_4BIT,
        # token = "hf_...",  # Add if using private model
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=64,                     # Rank
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
            "gate_proj", "up_proj", "down_proj",     # MLP
        ],
        lora_alpha=32,            # Scaling factor
        lora_dropout=0.05,        # Dropout for regularization
        bias="none",              # No bias
        use_gradient_checkpointing="unsloth",  # Save memory
        random_state=42,
        use_rslora=False,         # Disable rank-stabilized LoRA (not needed)
    )

    return model, tokenizer


def main():

    qa_data = load_qa_dataset("./data/qa_dataset.jsonl")
    formatted_texts = format_qa_for_training(qa_data)

    model, tokenizer = load_model_and_tokenizer()
    config = load_training_config()
    MAX_SEQ_LENGTH = config["max_seq_length"]
    OUTPUT_DIR = config["output_dir"]
    GRAD_ACCUM_STEPS = config["grad_accum_steps"]
    BATCH_SIZE = config["batch_size"]
    EPOCHS = config["epochs"]
    LEARNING_RATE = config["learning_rate"]
    USE_BF16 = config["use_bf16"]

    def tokenize_and_shift_labels(examples):
        """Tokenize and shift labels in one pass to avoid list/tensor issues."""
        # Tokenize
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt",
        )

        # Shift labels for causal LM
        labels = tokenized["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100
        labels = labels.roll(shifts=-1, dims=1)
        labels[:, -1] = -100

        # Return as dict (Hugging Face Dataset expects this format)
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels,
        }

    # Create dataset and tokenize in one step
    dataset = Dataset.from_dict({"text": formatted_texts})
    tokenized_dataset = dataset.map(
        tokenize_and_shift_labels,
        batched=True,
        remove_columns=["text"],
        batch_size=BATCH_SIZE,  # Process in batches to save memory
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        optim="adamw_8bit",       # Memory-efficient optimizer
        bf16=USE_BF16,            # Better precision
        tf32=True,                # TensorFloat-32 for faster training (if supported)
        save_steps=50,
        logging_steps=5,
        report_to="none",         # Disable WandB to save memory
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        # unsloth2-specific optimizations
        ddp_find_unused_parameters=False,  # Avoid DDP issues
        fp16=False,               # Disable fp16 (use bf16 instead)
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)    


if __name__ == "__main__":
    main()