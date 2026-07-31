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

### Dataset creation with emails download

From the project's root directory.

```bash
python -m run.compile_dataset.extract_from_emails
```

If you want to use cython pre-compilation of the utilities extracting data from the emails, 
you need to build cython from inside `utils/`

```bash
python setup_html_parsers.py build_ext --inplace
```

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

### Q&A with ProgramAsWeights (PAW)

This repository also includes a Q&A system using [ProgramAsWeights](https://programasweights.com/) (PAW), which compiles natural language specifications into neural programs that run locally via llama.cpp.

**Features:**
- Two approaches: pure PAW with examples, and hybrid PAW + CSV lookup
- Automatically extracts run numbers and fields from questions
- Falls back to CSV lookup when PAW doesn't have the answer
- Cache management for forcing recompilation

**Setup:**

```bash
# Install PAW dependency
pip install programasweights --extra-index-url https://pypi.programasweights.com/simple/
```

**Important:** You need a PAW API key. Get it from https://programasweights.com/settings and add to `config/.env.local`:
```
PAW_API_KEY="your_key_here"
```

**Troubleshooting: Illegal Instruction Error**

If you encounter `Illegal instruction (core dumped)` when running PAW, this is due to CPU architecture incompatibility with the pre-built `llama-cpp-python` wheels. This commonly affects newer Intel CPUs (e.g., Meteor Lake / Core Ultra series).

**Solution:** Rebuild `llama-cpp-python` from source with CPU-compatible flags:

```bash
# Uninstall the pre-built wheel
pip uninstall -y llama-cpp-python

# Rebuild from source with compatible CPU flags (AVX2 + FMA, no AVX-512)
FORCE_CMAKE=1 CMAKE_ARGS="-DLLAMA_AVX2=ON -DLLAMA_FMA=ON -DLLAMA_BLAS=OFF -DLLAMAAVX512=OFF" \
pip install --no-cache-dir --force-reinstall --no-binary=llama-cpp-python llama-cpp-python==0.3.19
```

This disables AVX-512 (not supported on Intel Core Ultra) while keeping AVX2 and FMA optimizations for better performance.

**Usage:**

```bash
# Run interactive Q&A
python run/retrieve_information/qa_w_paw.py

# Force recompilation of PAW programs (after changing code)
python run/retrieve_information/qa_w_paw.py --force-recompile

# Clear all PAW cache
python run/retrieve_information/qa_w_paw.py --clear-all
```

**Example questions:**
- "What trigger mode was used in run #635?"
- "Who conducted run #1234?"
- "What setup was used for run #5678?"

**Note:** On first run, programs will be compiled and downloaded (~2-3 minutes). Subsequent runs use the local cache.

**Cache Management:**

If you modify the compilation functions (e.g., `compile_pure_paw_with_examples()`), you need to clear the cache for changes to take effect:

```bash
# Option 1: Force recompile all programs
python run/retrieve_information/qa_w_paw.py --force-recompile

# Option 2: Manually delete specific program cache
# List programs:
ls ~/.cache/programasweights/programs/
# Delete specific program:
rm -rf ~/.cache/programasweights/programs/<program_id>/

# Option 3: Clear everything
rm -rf ~/.cache/programasweights/
```

The PAW implementation tracks program IDs automatically, so `--force-recompile` will clear the correct cached programs.