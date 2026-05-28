# NectarTRON

## Overview

The goal of this repo is to fine-tune a language model on a Q&A dataset built on NectarCAM data. As a starting point, we will consider Q&A generated from NectarCAM ELOGs, which include information about testing runs taken with NectarCAM cameras in the dark rooms.
The fine-tuning is run with [Unsloth](https://unsloth.ai/).

**Why Unsloth?**
- Unsloth is specifically designed for fast fine-tuning with limited VRAM
- Unsloth uses 4-bit quantization + LoRA to fit large models on small GPUs

## Hardware Setup

The fine-tuning tests have been completed with the following hardware:
- ✓ NVIDIA RTX PRO 2000 Blackwell (8GB VRAM)
- ✓ Intel Core Ultra 9, 16 cores, 32GB RAM
- ✓ ~30GB disk space for model + output

## Setup Instructions

### 1. Create Virtual Environment

```bash
# Create fresh venv
python3 -m venv .venv

# Activate
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install fine-tuning dependencies
pip install -r requirements-finetune.txt
```

**Note:** This may take 5-10 minutes. If you hit issues with `bitsandbytes` on Linux, you might need:

```bash
pip install bitsandbytes-cuda120  # For CUDA 12.0
```