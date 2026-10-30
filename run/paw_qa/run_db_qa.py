#!/usr/bin/env python3
"""
Usage:
    python -m run.paw_qa.run_db_qa
"""

import argparse
import re

import pandas as pd

import programasweights as paw

from utils.clean_data import render_context

# --- MUST match make_training_set.py exactly ---------------------------
CONTEXT_PREFIX = "Context:"
QUESTION_PREFIX = "Question:"
EXCLUDED_COLUMNS = {"Message ID"}
MAX_FIELD_CHARS = 300


# TODO: check and use this

# -----------------------------------------------------------------------------

RUN_ID_RE = re.compile(r"run\s*#?\s*(\d+)", re.IGNORECASE)
TOKEN_RE = re.compile(r"[a-z0-9]+")
SEARCH_STOPWORDS = {
    "what", "when", "where", "which", "who", "whose", "whom", "how",
    "was", "were", "the", "for", "from", "about", "give", "tell",
    "does", "did", "this", "that", "with", "run", "runs", "used",
    "please", "many", "much",
}


class RunDatabase:
    """Retrieval layer over the runs CSV. Reads with pandas exactly like
    make_training_set.py, so cell rendering is identical. Swap for SQL when the lookup
    logic outgrows a linear scan."""

    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        if df.empty:
            raise SystemExit(f"{csv_path}: no rows found")
        self.rows = [row for _, row in df.iterrows()]
        self._common_tokens = None
        print(f"Loaded {len(self.rows)} runs")

    @staticmethod
    def _tokens(text: str) -> set:
        return set(TOKEN_RE.findall(text.lower()))

    def common_tokens(self) -> set:
        """Terms present in every row (lazily computed) — non-discriminative."""
        if self._common_tokens is None:
            rows = iter(self.rows)
            common = self._tokens(render_context(next(rows)))
            for row in rows:
                common &= self._tokens(render_context(row))
            self._common_tokens = common
        return self._common_tokens

    def find_run(self, run_id: int):
        """Row whose any cell mentions this run number ('Run #N: ...')."""
        for row in self.rows:
            for value in row:
                m = RUN_ID_RE.search(str(value))
                if m and int(m.group(1)) == run_id:
                    return row
        return None

    def search(self, terms, max_rows=3):
        """Rows whose context contains ALL terms (case-insensitive)."""
        hits = []
        for row in self.rows:
            text = render_context(row).lower()
            if all(t in text for t in terms):
                hits.append(row)
                if len(hits) >= max_rows:
                    break
        return hits

    def retrieve(self, question: str, max_rows=3):
        """Best-effort retrieval: run-ID match first, keyword search second."""
        m = RUN_ID_RE.search(question)
        if m:
            row = self.find_run(int(m.group(1)))
            if row is not None:
                return [row]
        common = self.common_tokens()
        tokens = [
            t for t in TOKEN_RE.findall(question.lower())
            if len(t) >= 4 and t not in SEARCH_STOPWORDS and t not in common
        ]
        if tokens:
            return self.search(tokens, max_rows=max_rows)
        return []


class PawRunQA:
    """Retrieval + PAW program call. Loads the program once, then serves."""

    def __init__(self, csv_path, program_id, offline=True):
        self.db = RunDatabase(csv_path)
        self.fn = paw.function(program_id, offline=offline)
        self.max_rows = 3

    def ask(self, question: str):
        rows = self.db.retrieve(question, max_rows=self.max_rows)
        if not rows:
            return "I could not find a matching run in the database."
        context = "\n".join(render_context(r) for r in rows)
        user_input = f"{CONTEXT_PREFIX} {context}\n{QUESTION_PREFIX} {question.strip()}"
        return self.fn(user_input)


def qa_main():
    parser = argparse.ArgumentParser(
        description="Ask questions about your runs database via the PAW program")
    parser.add_argument("--csv", required=False, 
                        default="./data/extracted_entries.csv",
                        help="Database CSV, one row per run")
    parser.add_argument("--program-id", required=False, 
                        default="e10d624883881b775f75", 
                        help="PAW program ID (swapped adapter)")
    parser.add_argument("--online", action="store_true",
                        help="Allow network access (default: fully offline)")
    args = parser.parse_args()

    qa = PawRunQA(args.csv, args.program_id, offline=not args.online)
    print("Interactive QA — empty line or Ctrl-C/Ctrl-D to quit.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        print(qa.ask(question))

if __name__ == "__main__":
    
    qa_main()