import pandas as pd
import json
import re


from utils.logging_config import setup_logger, log_and_print

logger = setup_logger(
    log_file_name="create_dataset",
    logger_name="nectartron",
)


def extract_run_number(keyword):
    """Extract run number from Keyword field."""
    match = re.search(r'Run\s+#?(\d+)', keyword, re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def extract_subject(keyword):
    """Extract subject from Keyword field."""
    match = re.search(r'Subject:\s*(.+?)(?:\s+|$)', keyword)
    if match:
        return match.group(1).strip()
    return keyword


def create_qa_pairs(row):
    """Create question-answer pairs from a data row."""
    qa_pairs = []
    
    run_num = extract_run_number(row['Keyword'])
    subject = extract_subject(row['Keyword'])
    entry_time = row['Entry time']
    author = row['Author']
    setup = row['Setup']
    category = row['Category']
    module_count = row['ModuleCount']
    trigger_modes = row['TriggerModes']
    light_source = row['LightSource']
    
    # Only create QA pairs if we have a run number
    if not run_num:
        return qa_pairs
    
    # Q&A about run date
    if entry_time:
        qa_pairs.append({
            "question": f"What run was conducted on {entry_time}?",
            "answer": f"Run #{run_num} was conducted on {entry_time}."
        })
        qa_pairs.append({
            "question": f"When was run #{run_num} taken?",
            "answer": f"Run #{run_num} was taken on {entry_time}."
        })
    
    # Q&A about who conducted the run
    if author:
        qa_pairs.append({
            "question": f"Who conducted run #{run_num}?",
            "answer": f"Run #{run_num} was conducted by {author}."
        })
    
    # Q&A about module count
    if module_count and str(module_count).strip():
        qa_pairs.append({
            "question": f"How many modules were used in run #{run_num}?",
            "answer": f"Run #{run_num} used {module_count} modules."
        })
    
    # Q&A about trigger mode
    if trigger_modes and str(trigger_modes).strip():
        qa_pairs.append({
            "question": f"What trigger mode was used in run #{run_num}?",
            "answer": f"Run #{run_num} used {trigger_modes} trigger mode."
        })
    
    # Q&A about light source
    if light_source and str(light_source).strip():
        qa_pairs.append({
            "question": f"What light source was used in run #{run_num}?",
            "answer": f"Run #{run_num} used {light_source}."
        })
    
    # Q&A about setup
    if setup and str(setup).strip():
        qa_pairs.append({
            "question": f"What setup was used for run #{run_num}?",
            "answer": f"Run #{run_num} was conducted with {setup} setup."
        })
    
    # Q&A about category
    if category and str(category).strip():
        qa_pairs.append({
            "question": f"What category does run #{run_num} belong to?",
            "answer": f"Run #{run_num} belongs to the {category} category."
        })
    
    # Q&A about run subject/purpose
    if subject:
        qa_pairs.append({
            "question": f"What is the purpose of run #{run_num}?",
            "answer": f"Run #{run_num}: {subject}"
        })
    
    return qa_pairs


def main():
    # Read CSV file
    df = pd.read_csv("./data/extracted_entries.csv")
    
    # Create dataset
    dataset = []
    for idx, row in df.iterrows():
        qa_pairs = create_qa_pairs(row)
        for qa in qa_pairs:
            dataset.append(qa)
    
    # Save as JSONL (one JSON object per line)
    output_file = "./data/qa_dataset.jsonl"
    with open(output_file, 'w') as f:
        for item in dataset:
            f.write(json.dumps(item) + '\n')
    
    # Also save as JSON for easier inspection
    output_json = "./data/qa_dataset.json"
    with open(output_json, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    log_and_print(logger, f"Dataset created with {len(dataset)} Q&A pairs")
    log_and_print(logger, f"Saved to {output_file} (JSONL) and {output_json} (JSON)")
    
    # Print first few examples
    log_and_print(logger, "\nFirst 5 Q&A pairs:")
    for i, qa in enumerate(dataset[:5]):
        log_and_print(logger, f"\n{i+1}. Q: {qa['question']}")
        log_and_print(logger, f"   A: {qa['answer']}")


if __name__ == "__main__":
    main()
