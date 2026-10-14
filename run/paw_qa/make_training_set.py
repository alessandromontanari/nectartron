# ===========================================================================
# make_training_set.py  
# (similar to compile_dataset/create_dataset.py, updated for context training)
#
# CHANGES from original version:
#   1. Every question now carries a Context block (the row rendered as text),
#      so the adapter learns to EXTRACT answers from records instead of
#      memorizing pairs -> generalizes to runs recorded after training.
#   2. NaN-safety: with pandas, `if entry_time:` is True even when the cell
#      is NaN (NaN is truthy in Python!). All field checks now go through
#      normalize_cell(), which returns "" for missing values.
#   3. render_context() here MUST stay byte-identical to the one in
#      run_db_qa.py: training and runtime must produce the SAME
#      context string. Change both together, then retrain.
#   4. Long fields (Content) are clipped to MAX_FIELD_CHARS to protect the
#      interpreter's 2048-token window; the purpose answer uses the SAME
#      clipped text, so the answer is always derivable from the context.
#
# Pipeline position:
#   data/extracted_entries.csv --> make_training_set.py
#     --> data/qa_context_dataset.jsonl --> train_paw_lora_adapter.py
# ===========================================================================

import pandas as pd
import json
import re

from utils.logging_config import setup_logger, log_and_print

logger = setup_logger(
    log_file_name="create_dataset",
    logger_name="nectartron",
)

# --- MUST match run_db_qa.py exactly -------------------------------
CONTEXT_PREFIX = "Context:"
QUESTION_PREFIX = "Question:"
EXCLUDED_COLUMNS = {"Message ID"}   # not useful for answering
MAX_FIELD_CHARS = 300               # protect the 2048-token runtime window


def normalize_cell(value) -> str:
    """Render a pandas cell as clean text; '' for NaN/empty.

    A numeric column with any missing value is read as float (e.g. 12.0);
    integral floats are rendered without the trailing '.0' so the model
    can copy the exact string from the context.
    """
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value).strip()


def clip(text: str, max_chars: int = MAX_FIELD_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"


def render_context(row) -> str:
    """One CSV row as a compact text record — MUST match run_db_qa.py exactly."""
    parts = []
    for key, value in row.items():
        if key in EXCLUDED_COLUMNS:
            continue
        text = normalize_cell(value)
        if not text or text.lower() in ("nan", "none", "null"):
            continue
        parts.append(f"{key}: {clip(text)}")
    return " | ".join(parts)
# -----------------------------------------------------------------------------


def extract_run_number(keyword):
    """Extract run number from Keyword field."""
    pattern = r'Run #([^:]+)'
    match = re.search(pattern, keyword, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def create_qa_pairs(row):
    """Create context-augmented question-answer pairs from a data row."""
    qa_pairs = []

    run_num = extract_run_number(row['Subject'])
    if not run_num:
        return qa_pairs

    context = render_context(row)
    if not context:
        return qa_pairs

    entry_time = normalize_cell(row['Entry time'])
    author = normalize_cell(row['Author'])
    setup = normalize_cell(row['Setup'])
    category = normalize_cell(row['Category'])
    module_count = normalize_cell(row['ModuleCount'])
    trigger_modes = normalize_cell(row['TriggerModes'])
    light_source = normalize_cell(row['LightSource'])
    content = clip(normalize_cell(row['Content']))

    def add(question, answer):
        qa_pairs.append({
            "question": f"{CONTEXT_PREFIX} {context}\n{QUESTION_PREFIX} {question}",
            "answer": answer,
        })

    # Q&A about run date
    if entry_time:
        add(f"What run was conducted on {entry_time}?",
            f"Run #{run_num} was conducted on {entry_time}.")
        add(f"When was run #{run_num} taken?",
            f"Run #{run_num} was taken on {entry_time}.")

    # Q&A about who conducted the run
    if author:
        add(f"Who conducted run #{run_num}?",
            f"Run #{run_num} was conducted by {author}.")

    # Q&A about module count
    if module_count:
        add(f"How many modules were used in run #{run_num}?",
            f"Run #{run_num} used {module_count} modules.")

    # Q&A about trigger mode
    if trigger_modes:
        add(f"What trigger mode was used in run #{run_num}?",
            f"Run #{run_num} used {trigger_modes} trigger mode.")

    # Q&A about light source
    if light_source:
        add(f"What light source was used in run #{run_num}?",
            f"Run #{run_num} used {light_source}.")

    # Q&A about setup
    if setup:
        add(f"What setup was used for run #{run_num}?",
            f"Run #{run_num} was conducted with {setup} setup.")

    # Q&A about category
    if category:
        add(f"What category does run #{run_num} belong to?",
            f"Run #{run_num} belongs to the {category} category.")

    # Q&A about run subject/purpose (same clipped text as in the context,
    # so the answer stays derivable from the context alone)
    if content:
        add(f"What is the purpose of run #{run_num}?",
            f"Run #{run_num}: {content}")

    return qa_pairs


def main():
    # Read CSV file
    df = pd.read_csv("./data/extracted_entries.csv")

    # Create dataset
    dataset = []
    for _, row in df.iterrows():
        dataset.extend(create_qa_pairs(row))

    # Save as JSONL (one JSON object per line)
    output_file = "./data/qa_context_dataset.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # Also save as JSON for easier inspection
    output_json = "./data/qa_context_dataset.json"
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    log_and_print(logger, f"Dataset created with {len(dataset)} Q&A pairs")
    log_and_print(logger, f"Saved to {output_file} (JSONL) and {output_json} (JSON)")

    # Print first few examples
    log_and_print(logger, "\nFirst 2 Q&A pairs:")
    for i, qa in enumerate(dataset[:2]):
        log_and_print(logger, f"\n{i+1}. Q: {qa['question']}")
        log_and_print(logger, f"   A: {qa['answer']}")


if __name__ == "__main__":
    main()