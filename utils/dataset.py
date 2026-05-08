import os
import json
from typing import Dict, List

# TODO: logging

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
                print(f"Warning: Skipping line {i+1} - JSON decode error: {e}")
    
    print(f"✓ Loaded {len(data)} Q&A pairs from {filepath}")
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
    
    return formatted_texts