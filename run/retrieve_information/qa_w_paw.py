import pandas as pd
import json
import re
import os
from typing import Optional

# Load environment variables from config/.env.local
from dotenv import load_dotenv
load_dotenv("/home/alessandro/Desktop/PostDocLLR/NectarCAM/nectarllm/config/.env.local")

from utils.dataset import load_data_for_paw
from utils.paw_conf import clear_paw_cache, force_recompile, compile_pure_paw_with_examples, compile_hybrid_parser

# Ensure PAW_API_KEY is set
if "PAW_API_KEY" not in os.environ or not os.environ["PAW_API_KEY"]:
    raise ValueError(
        "PAW_API_KEY not found in config/.env.local. "
        "Get your API key from https://programasweights.com/settings "
        "and add it to config/.env.local as: PAW_API_KEY=\"your_key_here\""
    )



# ============================================================================
# Unified Answering System
# ============================================================================

class NectarCAMQA:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.pure_paw_fn = compile_pure_paw_with_examples(df)
        self.question_parser = compile_hybrid_parser()

        # Field mapping: question keywords -> CSV columns
        self.field_mapping = {
            "trigger": "TriggerModes", "trigger_mode": "TriggerModes", "mode": "TriggerModes",
            "author": "Author", "who": "Author", "conducted": "Author",
            "setup": "Setup", "camera": "Setup",
            "light": "LightSource", "source": "LightSource",
            "date": "Entry time", "when": "Entry time", "time": "Entry time",
            "modules": "ModuleCount", "module": "ModuleCount", "count": "ModuleCount",
            "category": "Category", "type": "Category",
            "purpose": "Content", "details": "Content", "description": "Content",
        }

    # --- Pure PAW (with examples) ---
    def answer_with_pure_paw(self, question: str) -> Optional[str]:
        try:
            answer = self.pure_paw_fn(question)
            if answer and answer != "USE_CSV":
                return answer
            return None
        except Exception as e:
            print(f"[Pure PAW Error] {e}")
            return None

    # --- Hybrid PAW + CSV ---
    def answer_with_hybrid(self, question: str) -> Optional[str]:
        try:
            # Step 1: Parse with PAW
            parsed = self.question_parser(question)

            # Handle string output
            if isinstance(parsed, str):
                try:
                    parsed = json.loads(parsed)
                except json.JSONDecodeError:
                    parsed = {"run": None, "field": None}

            # Step 2: Extract run number
            run_num = parsed.get("run")
            if not run_num:
                # Fallback: regex extraction
                match = re.search(r"#(\d+(?:-\d+)*)", question) or re.search(r"\b(\d{4}(?:-\d+)?)", question)
                if match:
                    run_num = match.group(1)

            if not run_num:
                return None

            # Step 3: Determine field
            field = parsed.get("field")
            if not field:
                # Infer from question
                for keyword, col in self.field_mapping.items():
                    if keyword in question.lower():
                        field = keyword
                        break

            if not field:
                return None

            csv_field = self.field_mapping.get(field.lower())
            if not csv_field:
                return None

            # Step 4: CSV Lookup
            # Handle range queries like #1234-1235
            if "-" in str(run_num):
                run_numbers = str(run_num).split("-")
                run_data = self.df[
                    self.df["RunNumber"].apply(
                        lambda x: any(rn in str(x) for rn in run_numbers)
                    )
                ]
            else:
                run_data = self.df[self.df["RunNumber"] == run_num]

            if not run_data.empty:
                return str(run_data[csv_field].iloc[0])

            return None

        except Exception as e:
            print(f"[Hybrid Error] {e}")
            return None

    # --- Unified Answer ---
    def answer(self, question: str, method: str = "hybrid") -> str:
        question = question.strip()
        if method == "pure":
            # TODO: pure does not seem to work very well, 
            # as if only a limited number of examples are passed, 
            # then the model is also limited in its answers
            return self.answer_with_pure_paw(question) or "Not found."
        elif method == "hybrid":
            return self.answer_with_hybrid(question) or "Not found."
        else:  # auto
            # Try pure first, then hybrid
            answer = self.answer_with_pure_paw(question)
            if answer:
                return answer
            return self.answer_with_hybrid(question) or "Not found."

    # --- Interactive Mode ---
    def interactive(self):
        print("\n=== NectarCAM Q&A ===")
        print("Ask about camera runs (e.g., 'What trigger mode was used in run #635?')")
        print("Type 'quit' to exit.\n")
        while True:
            question = input("Q: ").strip()
            if question.lower() in ("quit", "exit", "q"):
                break
            if question:
                answer = self.answer(question)
                print(f"A: {answer}\n")


# ============================================================================
# Main
# ============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="NectarCAM Q&A with PAW")
    parser.add_argument(
        "--force-recompile", "-f",
        action="store_true",
        help="Force recompilation of PAW programs by clearing cache"
    )
    parser.add_argument(
        "--clear-all",
        action="store_true",
        help="Clear all PAW cache (including base models)"
    )
    args = parser.parse_args()
    
    if args.clear_all:
        clear_paw_cache(all=True)
        print("All PAW cache cleared.")
        return
    
    if args.force_recompile:
        force_recompile()
        print("PAW program caches cleared. Programs will recompile on next run.")
        return
    
    df = load_data_for_paw()
    qa = NectarCAMQA(df=df)
    qa.interactive()


if __name__ == "__main__":
    main()