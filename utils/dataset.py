import os
import json
from pathlib import Path
from typing import Dict, List
import logging
logger = logging.getLogger(__name__)

import pandas as pd

from utils.logging_config import log_and_print

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