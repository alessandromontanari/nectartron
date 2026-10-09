import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset
import logging
logger = logging.getLogger(__name__)

import pandas as pd

from utils.logging_config import log_and_print

MAX_LEN = 1024
QUESTION_KEYS = ["question", "input", "q", "prompt", "query"]
ANSWER_KEYS = ["answer", "output", "a", "response", "completion"]

BASE_DIR = Path(__file__).parent.parent.parent  # nectarllm directory
DATA_PATH = str(BASE_DIR / "data" / "extracted_entries.csv")

# TODO: adapt prompt text depending on the model 
# (e.g., some models might require different special tokens or formatting)

def load_qa_dataset(filepath: str) -> List[Dict[str, str]]:
    """Load Q&A dataset from JSONL file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")
    
    data = []
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            try:
                qa = json.loads(line)
                data.append({
                    "question": qa.get("question", ""),
                    "answer": qa.get("answer", "")
                })
            except json.JSONDecodeError as e:
                log_and_print(logger, f"Skipping line {i+1} - JSON decode error: {e}", level="warning")
    
    log_and_print(logger, f"✓ Loaded {len(data)} Q&A pairs from {filepath}")
    return data


def format_qa_for_training(qa_list: List[Dict[str, str]]) -> List[str]:
    """Format Q&A pairs into training format with special tokens."""
    formatted_texts = []
    
    for qa in qa_list:
        question = qa['question'].strip()
        answer = qa['answer'].strip()
        
        # Format: <s>[INST] question [/INST] answer </s>
        text = f"<s>[INST] {question} [/INST] {answer} </s>"
        formatted_texts.append(text)

    log_and_print(logger, f"✓ Formatted {len(formatted_texts)} Q&A pairs for training")
    return formatted_texts


def load_data_for_paw() -> pd.DataFrame:
    """Load and preprocess the extracted entries for PAW Q&A."""
    df = pd.read_csv(DATA_PATH)
    df = df.fillna("")
    df.columns = [col.strip() for col in df.columns]
    # Extract run numbers (handles formats like #1234 or #1234-1235)
    df["RunNumber"] = df["Subject"].str.extract(r"#(\d+(?:-\d+)*)")[0]
    # Also try to extract from Content if Subject is empty
    df.loc[df["RunNumber"].isna(), "RunNumber"] = df["Content"].str.extract(r"#(\d+(?:-\d+)*)")[0]
    return df


def pick_key(record: Dict, keys: List[str]) -> Optional[str]:
    for k in keys:
        if k in record and isinstance(record[k], str) and record[k].strip():
            return k
    return None


def load_qa_pairs(path: str) -> List[Tuple[str, str]]:
    """Load (question, answer) pairs from a JSONL file, tolerating common key names."""
    pairs: List[Tuple[str, str]] = []
    skipped = 0
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            qk, ak = pick_key(record, QUESTION_KEYS), pick_key(record, ANSWER_KEYS)
            if not qk or not ak:
                skipped += 1
                print(f"  ! line {line_no}: no recognizable Q/A keys, skipping")
                continue
            pairs.append((record[qk].strip(), record[ak].strip()))
    if skipped:
        print(f"  (skipped {skipped} unusable lines)")
    if not pairs:
        raise SystemExit(f"No usable Q/A pairs found in {path}")
    return pairs


class QaDataset(Dataset):
    """Tokenized prompt+answer pairs with labels masked on the prompt part
    (completion-only loss: the model is only trained to produce answers)."""

    def __init__(self, pairs, tokenizer, prompt_fn, max_len=MAX_LEN):
        self.examples = []
        for question, answer in pairs:
            prompt_text = prompt_fn(question)
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
            input_ids = prompt_ids + answer_ids
            if len(input_ids) > max_len:
                input_ids = input_ids[:max_len]
            labels = [-100] * min(len(prompt_ids), len(input_ids)) + input_ids[len(prompt_ids):]
            self.examples.append({"input_ids": input_ids, "labels": labels})

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate(batch, pad_token_id):
    """Right-pad input_ids with pad_token, labels with -100."""
    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        pad = maxlen - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_token_id] * pad)
        labels.append(b["labels"] + [-100] * pad)
        attn.append([1] * len(b["input_ids"]) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
    }