# ===========================================================================
# the_maestro.py, orchestrating the whole pipeline
#
# One command for the recurring loop:
#   (optional) regenerate dataset
#   -> train LoRA adapter
#   -> convert PEFT adapter to GGUF
#   -> VALIDATE the GGUF (magic bytes) before touching the cache
#   -> back up the previous adapter (timestamped)
#   -> swap adapter.gguf into the PAW program dir
#   -> drop the stale prefix KV cache
#   -> (optional) offline smoke test
#
# Run it with your project venv's python:
#   python -m run.paw_qa.the_maestro \
#       --program-id <id> \
#       --make-dataset \
#       --smoke-test "Who conducted run #325?"
#
#   # Convert + swap only (skip retraining):
#   python -m run.paw_qa.the_maestro --program-id <id> --skip-train
# ===========================================================================

import argparse
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

from utils.logging_config import setup_logger, log_and_print

TRAIN_MODULE = "run.paw_qa.train_paw_lora_adapter"   # adjust to your repo layout
# in run/paw_qa/train_paw_lora_adapter.py
DATASET_MODULE = "run.paw_qa.make_training_set"      # used only with --make-dataset
# in run/paw_qa/make_training_set.py
CONVERTER_DIR = "tools/llamacpp-lora-converter"      # vendored llama.cpp converter
# in tools/llamacpp-lora-converter.py

logger = setup_logger(
    log_file_name="train_paw_lora_adapter",
    logger_name="nectartron",
)


def programs_dir() -> Path:
    return Path.home() / ".cache" / "programasweights" / "programs"


def run_step(title: str, cmd, cwd=None):
    cmd = [str(c) for c in cmd]
    log_and_print(logger, f"\n=== {title} ===\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    if result.returncode != 0:
        raise SystemExit(
            f"\n✗ {title} failed (exit code {result.returncode}).\n"
            "  Pipeline stopped BEFORE touching the PAW cache — nothing was swapped.")
    log_and_print(logger, f"✓ {title}")


def validate_gguf(path: Path):
    """Refuse to swap in a broken file: check the GGUF magic bytes."""
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise SystemExit(f"✗ {path} is not a valid GGUF file — refusing to touch the PAW cache.")


def main():
    parser = argparse.ArgumentParser(
        description="Retrain/convert/swap the PAW QA adapter in one shot")
    parser.add_argument("--program-id", default="e10d624883881b775f75",
                        help="PAW program ID to retrain for (same one you compiled)")
    # default program ID is for the following spec:
    # 'Answer questions about the current run: its trigger mode, 
    # who conducted it, its date, and similar run properties.'
    # This is the program compiled by run/paw/qa/compile_paw_program.py, 
    # should you change it, then you may want to change the default ID too
    parser.add_argument("--data", default="./data/qa_context_dataset.jsonl",
                        help="Training JSONL (context-augmented)")
    parser.add_argument("--adapter-out", default="./output_adapters/qa_adapter",
                        help="PEFT adapter output dir")
    parser.add_argument("--gguf", default="./output_adapters/adapter.gguf",
                        help="Converted GGUF output path")
    parser.add_argument("--outtype", default="q8_0",
                        choices=["f32", "f16", "bf16", "q8_0", "auto"],
                        help="GGUF LoRA precision (rank-16 B tensors fall back to F16)")
    parser.add_argument("--make-dataset", action="store_true",
                        help="Regenerate qa_dataset.jsonl first")
    parser.add_argument("--skip-train", action="store_true",
                        help="Convert + swap only (adapter already trained)")
    parser.add_argument("--extra-train-args", default="",
                        help='Extra args for the train module, e.g. "--epochs 4 --lr 2e-4"')
    parser.add_argument("--no-backup", action="store_true",
                        help="Do not keep a timestamped copy of the previous adapter")
    parser.add_argument("--smoke-test", default=None,
                        help="Offline question to ask after the swap, e.g. --smoke-test \"Who conducted run #325?\"")
    args = parser.parse_args()

    # ----- locate the compiled program in the PAW cache -----
    program_dir = programs_dir() / args.program_id
    if not (program_dir / "prompt_template.txt").is_file():
        raise SystemExit(
            f"✗ No cached program at {program_dir}.\n"
            "  Compile and LOAD it once first (paw.compile_and_load) — "
            "the cache dir is created at load time.")

    # ----- 1) dataset (optional) -----
    if args.make_dataset:
        run_step("Regenerate dataset", [sys.executable, "-m", DATASET_MODULE])

    # ----- 2) train (optional) -----
    if not args.skip_train:
        train_cmd = [
            sys.executable, "-m", TRAIN_MODULE,
            "--data", args.data,
            "--program-dir", program_dir,
            "--adapter-out", args.adapter_out,
        ] + (shlex.split(args.extra_train_args) if args.extra_train_args else [])
        run_step("Train LoRA adapter", train_cmd)

    # ----- 3) convert -----
    gguf = Path(args.gguf)
    gguf.parent.mkdir(parents=True, exist_ok=True)
    run_step(
        "Convert LoRA to GGUF",
        [sys.executable, "convert_lora_to_gguf.py",
         "--outfile", gguf.resolve(), "--outtype", args.outtype,
         Path(args.adapter_out).resolve()],
        cwd=CONVERTER_DIR,
    )
    validate_gguf(gguf)

    # ----- 4) swap + drop stale prefix cache -----
    target = program_dir / "adapter.gguf"
    prefix_cache = program_dir / "prefix_kv_cache.bin"
    log_and_print(logger, f"\n=== Swap adapter into {program_dir} ===")
    if not args.no_backup and target.exists():
        backup = program_dir / f"adapter.gguf.bak-{time.strftime('%Y%m%d-%H%M%S')}"
        shutil.copy2(target, backup)
        log_and_print(logger, f"  previous adapter backed up: {backup.name}")
    shutil.copy2(gguf, target)
    prefix_cache.unlink(missing_ok=True)
    log_and_print(logger, "  ✓ new adapter in place; stale prefix KV cache dropped")

    # ----- 5) smoke test (optional, fully offline) -----
    if args.smoke_test:
        log_and_print(logger, "\n=== Offline smoke test ===")
        import programasweights as paw
        fn = paw.function(args.program_id, offline=True)
        log_and_print(logger, f"  Q: {args.smoke_test}")
        log_and_print(logger, f"  A: {fn(args.smoke_test)}")

    log_and_print(logger, f"\n✅ Redeploy complete: program {args.program_id} now runs {gguf}")

# Entry point for the copied-out file:
if __name__ == "__main__": 
    main()