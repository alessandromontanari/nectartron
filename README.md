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

### 3. Verify GPU Access

```bash
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'GPU available: {torch.cuda.is_available()}'); print(f'GPU name: {torch.cuda.get_device_name(0)}')"
```

Should output something like:
```
PyTorch version: 2.X.X
GPU available: True
GPU name: NVIDIA RTX PRO 2000 Blackwell Generation Laptop GPU
```

## Usage

### Fine-tuning

From the project's root directory. 

```bash
python -m run.fine_tune
```

### Q&A with the model

From the project's root directory. 

```bash
python -m run.qa_w_model
```

One can change the question for the model inside the script.