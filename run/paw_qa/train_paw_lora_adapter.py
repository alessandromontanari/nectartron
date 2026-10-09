#!/usr/bin/env python3
"""
train_paw_lora_adapter.py — Train a LoRA adapter locally on private QA pairs,
for use as the adapter inside a PAW program directory.

Pipeline:
  1. Load qa_dataset.jsonl  ({"question": ..., "answer": ...} or {"input": ..., "output": ...})
  2. Render each pair in the SAME prompt format the PAW runtime will use
     (reads prompt_template.txt from your compiled program dir if available),
     so the adapter learns behavior in its execution context.
  3. LoRA fine-tune Qwen3-0.6B (the PAW interpreter) with completion-only loss
     (loss on the answer tokens, not the prompt).
  4. Save a standard PEFT adapter directory — ready for llama.cpp conversion.

Everything runs on your machine: the JSONL, the CSV it came from, and the
trained weights never leave this computer.

Usage:
    # If you already compiled a program via PAW (recommended, so the prompt
    # template matches execution time exactly):
    python train_paw_lora_adapter.py

Requirements (Blackwell GPUs need a CUDA 12.8 build of PyTorch):
    pip install torch --index-url https://download.pytorch.org/whl/cu128
    pip install transformers peft datasets accelerate

Hardware: defaults tuned for 8 GB VRAM (e.g. RTX PRO 2000 Blackwell):
    micro-batch 2 + gradient checkpointing. Peak memory is dominated by the
    fp32 logit tensors of Qwen3's ~152k-token vocab (~1.2 GB per 1024-token
    sequence), NOT by model or adapter weights. If still OOM:
    use --micro-bs 1 and/or --max-len 512.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import math
import random
import torch
from peft import LoraConfig, get_peft_model

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer
)
from programasweights import cache

from utils.dataset import load_qa_pairs, QaDataset, collate

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

MAX_LEN = 1024
VAL_FRACTION = 0.1      # held-out split to monitor overfitting
BASE_MODEL = "Qwen/Qwen3-0.6B"  
# the PAW interpreter — must match the runtime's base GGUF revision
SEED = 42

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
# Target modules: same set PAW itself trains (all attention + MLP projections).
# These names are what llama.cpp's convert_lora_to_gguf.py expects to find.

# ---------------------------------------------------------------------------
# Prompt rendering — must match how the PAW runtime will execute your program
# ---------------------------------------------------------------------------
def make_prompt_fn(program_dir, tokenizer):
    """
    If a PAW program dir is given, use its prompt_template.txt (the file with
    {INPUT_N} placeholders) — this is exactly what the runtime does at inference.
    Otherwise fall back to the Qwen3 chat template with thinking disabled
    (mirroring the PAW runtime's enable_thinking=False behavior).
    """
    template_path = Path(program_dir) / "prompt_template.txt" if program_dir else None

    if template_path and template_path.exists():
        template = template_path.read_text(encoding="utf-8")
        print(f"Using PAW prompt template from {template_path}")

        def prompt_fn(question: str) -> str:
            # Current PAW programs use {INPUT_PLACEHOLDER}; some older formats
            # use {INPUT_0}. str.replace (not .format): the template may contain
            # literal braces (e.g. JSON examples) that would crash .format().
            for placeholder in ("{INPUT_PLACEHOLDER}", "{INPUT_0}"):
                if placeholder in template:
                    return template.replace(placeholder, question)
            raise ValueError(
                "No input placeholder found in prompt_template.txt "
                "(expected {INPUT_PLACEHOLDER} or {INPUT_0})"
            )
        return prompt_fn

    print("No PAW program template found — using Qwen3 chat template (enable_thinking=False)")

    def prompt_fn(question: str) -> str:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # PAW runtime behavior for Qwen3
        )
    return prompt_fn


def check_gpu() -> bool:
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        print(f"GPU: {torch.cuda.get_device_name(0)} (compute capability {cap[0]}.{cap[1]})")
        return True
    print("⚠  No usable CUDA device detected.")
    print("   If you have a Blackwell GPU (RTX PRO 2000), your PyTorch build predates")
    print("   CUDA 12.8 support, and the GPU is invisible to it. Reinstall with:")
    print("       pip install torch --index-url https://download.pytorch.org/whl/cu128")
    print("   Falling back to CPU — training will be very slow.")
    return False


def train_adapter(args, program_dir, train_pairs, val_pairs):

    # ----- model -----
    print(f"Loading {args.base_model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else (
        torch.float16 if torch.cuda.is_available() else torch.float32
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=dtype,  # bf16: natively supported on Blackwell
        attn_implementation="sdpa",  # efficient enough at 0.6B scale
    )
    model.config.use_cache = False  # incompatible with LoRA training

    # ----- LoRA -----
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=2 * args.rank,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,  # must match what llama.cpp can convert
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()    

    # ----- datasets -----
    prompt_fn = make_prompt_fn(program_dir, tokenizer)
    train_ds = QaDataset(train_pairs, tokenizer, prompt_fn, max_len=args.max_len)
    val_ds = QaDataset(val_pairs, tokenizer, prompt_fn) if val_pairs else None

    # ----- training args -----
    grad_accum = max(1, math.ceil(args.batch_size / args.micro_bs))
    effective_batch = args.micro_bs * grad_accum
    steps_per_epoch = max(1, math.ceil(len(train_ds) / effective_batch))
    
    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.micro_bs,
        gradient_accumulation_steps=grad_accum,
        per_device_eval_batch_size=args.micro_bs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        eval_strategy="steps" if val_ds else "no",
        eval_steps=max(1, steps_per_epoch // 2),
        save_strategy="no",           # we save the final adapter ourselves
        gradient_checkpointing=not args.no_grad_ckpt,  # keeps activations small for 8 GB VRAM
        # use_reentrant=False is REQUIRED with PEFT — otherwise gradients don't flow
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        report_to="none",
        seed=SEED,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=lambda batch: collate(batch, tokenizer.pad_token_id),
    )

    print("\n=== Training ===")
    trainer.train()

    # ----- save PEFT adapter -----
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))       # adapter_config.json + adapter_model.safetensors
    tokenizer.save_pretrained(str(out_dir))   # convenience; not needed by llama.cpp
    print(f"\n✅ Adapter saved to {out_dir}")

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Trainable LoRA parameters: {total:,}")



def main():
    parser = argparse.ArgumentParser(description="Local LoRA training for a PAW program adapter")
    parser.add_argument("--data", default="./data/qa_dataset.jsonl", help="Path to qa_dataset.jsonl")
    parser.add_argument("--out", default="./output_adapters/qa_adapter", help="Output PEFT adapter directory (e.g. ./out/qa_adapter)")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--epochs", type=float, default=3.0, help="Epochs (small QA sets overfit fast; 2-4 is typical)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank (PAW uses 8; 16-32 gives more capacity for QA memorization)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Effective batch size (reached via micro-batch accumulation)")
    parser.add_argument("--micro-bs", type=int, default=2,
                        help="Batch size per forward pass. Memory is dominated by fp32 logits "
                             "of Qwen3's ~152k vocab (~1.2 GB per 1024-token sequence); "
                             "2 fits 8 GB VRAM, drop to 1 if OOM")
    parser.add_argument("--max-len", type=int, default=MAX_LEN,
                        help="Max tokens per example. Lowering to 512 shrinks the logits "
                             "tensor proportionally if your template + answers are short")
    parser.add_argument("--no-grad-ckpt", action="store_true",
                        help="Disable gradient checkpointing (faster, needs more VRAM)")
    args = parser.parse_args()

    random.seed(SEED)
    torch.manual_seed(SEED)
    check_gpu()

    print(f"Loading {args.data} ...")
    pairs = load_qa_pairs(args.data)
    print(f"  {len(pairs)} Q/A pairs loaded")

    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * VAL_FRACTION)) if len(pairs) >= 20 else 0
    val_pairs, train_pairs = pairs[:n_val], pairs[n_val:]
    print(f"  train={len(train_pairs)}  val={len(val_pairs)}")

    print("These are the available PAW programs in the cache:")
    for ii, p in enumerate(cache.list_cached_programs()):
        print(f"{ii}: {p['program_id']}  ->  {p['spec']}", end="\n")
    choice = int(
        input("Please choose a PAW program to train an adapter for (enter the index): ")
    )

    chosen_program = cache.list_cached_programs()[choice]
    program_dir = chosen_program['program_dir']

    train_adapter(args, program_dir, train_pairs, val_pairs)






if __name__ == "__main__":
    main()