import json
import pandas as pd
import programasweights as paw
import shutil
from pathlib import Path

PAW_CACHE_DIR = Path.home() / ".cache" / "programasweights"

# ============================================================================
# Program ID tracking (for cache management)
# ============================================================================
# Store program IDs so we can clear them later
_PROGRAM_IDS = {
    "pure_paw": None,
    "hybrid_parser": None,
}

def get_program_id(name: str) -> str | None:
    """Get the program ID for a named program."""
    return _PROGRAM_IDS.get(name)


# ============================================================================
# Cache management
# ============================================================================

def clear_paw_cache(slug: str = None, program_id: str = None, all: bool = False, 
                     spec_contains: str = None) -> bool:
    """Clear PAW cache to force recompilation.
    
    Parameters
    ----------
    slug: str
        Clear cache for a specific slug (e.g., 'nectarcam-qa-examples')
    program_id: str 
        Clear cache for a specific program ID
    all: bool
        Clear all PAW cache if True
    spec_contains: str 
        Clear programs whose spec contains this string
    
    Returns
    -------
    bool
        True if cache was cleared, False otherwise
    """

    if not PAW_CACHE_DIR.exists():
        return False
    
    cleared = False
    programs_dir = PAW_CACHE_DIR / "programs"
    
    # Clear all cache
    if all:
        if PAW_CACHE_DIR.exists():
            shutil.rmtree(PAW_CACHE_DIR)
            cleared = True
        return cleared
    
    # Clear specific program by ID
    if program_id:
        program_dir = programs_dir / program_id
        if program_dir.exists():
            shutil.rmtree(program_dir)
            cleared = True
    
    # Clear by slug - need to resolve program_id from slug_cache.json
    if slug:
        slug_cache = PAW_CACHE_DIR / "slug_cache.json"
        if slug_cache.exists():
            with open(slug_cache, "r") as f:
                cache_data = json.load(f)
            if slug in cache_data:
                program_id = cache_data[slug]
                program_dir = programs_dir / program_id
                if program_dir.exists():
                    shutil.rmtree(program_dir)
                    cleared = True
                # Remove from slug cache
                del cache_data[slug]
                with open(slug_cache, "w") as f:
                    json.dump(cache_data, f, indent=2)
    
    # Clear programs whose spec contains a specific string
    if spec_contains:
        if programs_dir.exists():
            for program_dir in programs_dir.iterdir():
                if program_dir.is_dir():
                    meta_file = program_dir / "meta.json"
                    if meta_file.exists():
                        try:
                            with open(meta_file, "r") as f:
                                meta = json.load(f)
                            if spec_contains in meta.get("spec", ""):
                                shutil.rmtree(program_dir)
                                cleared = True
                        except (json.JSONDecodeError, OSError):
                            pass
    
    return cleared


def force_recompile(slug: str = None) -> None:
    """
    Force recompilation of a PAW program by clearing its cache.
    
    Parameters
    ----------
    slug: str
        The slug of the program to recompile (e.g., 'nectarcam-qa-examples')
        If None, clears both main program caches.
    """
    if slug:
        clear_paw_cache(slug=slug)
    else:
        # Clear by program IDs if they were tracked
        for name, program_id in _PROGRAM_IDS.items():
            if program_id:
                clear_paw_cache(program_id=program_id)
        
        # Also try by slug in case they were saved
        clear_paw_cache(slug="nectarcam-qa-examples")
        clear_paw_cache(slug="nectarcam-question-parser")
        clear_paw_cache(slug="alessandromontanari/nectarcam-qa-examples")
        clear_paw_cache(slug="alessandromontanari/nectarcam-question-parser")
        
        # Also try by spec content
        clear_paw_cache(spec_contains="nectarcam-qa-examples")
        clear_paw_cache(spec_contains="nectarcam-question-parser")
        clear_paw_cache(spec_contains="NectarCAM camera runs")


# ============================================================================
# PAW Q&A pipelines
# ============================================================================

# ============================================================================
# Approach 1: Pure PAW with Examples (No Context Parameter)
# ============================================================================
def compile_pure_paw_with_examples(df: pd.DataFrame):
    """Compile PAW with representative examples from the dataset.
       PAW learns patterns from these examples.
    
    Returns
    -------
    paw.function
        A callable PawFunction
    """
    # # Generate examples from first 50 runs (to keep spec size manageable)
    # examples = []
    # # for _, row in df.head(50).iterrows():
    # for _, row in df.iterrows():
    #     run_num = row["RunNumber"]
    #     if pd.isna(run_num) or not run_num:
    #         continue

    #     examples.append(f"Q: What trigger mode was used in run #{run_num}? A: {row['TriggerModes']}")
    #     examples.append(f"Q: Who conducted run #{run_num}? A: {row['Author']}")
    #     examples.append(f"Q: What setup was used for run #{run_num}? A: {row['Setup']}")

    # examples_str = "\n".join(examples)

    # First compile to get the program object (which has the ID)
    program = paw.compile(
        spec=(
            "You are an expert on NectarCAM camera runs. "
            # "Answer questions based on the following examples:\n\n"
            # f"{examples_str}\n\n"
            "For questions about specific runs, extract the run number and the field requested. "
            "If the run number is not in the examples, return 'USE_CSV'. "
            "Otherwise, return the answer directly."
        ),
        slug="nectarcam-qa-examples",
        compiler="paw-4b-qwen3-0.6b",
        # compiler="paw-4b-gpt2",
    )
    
    # Store the program ID for cache management
    _PROGRAM_IDS["pure_paw"] = program.id
    
    # Now load the function
    qa_function = paw.function(program)
    return qa_function


# ============================================================================
# Approach 2: Hybrid PAW + CSV Lookup
# ============================================================================
def compile_hybrid_parser():
    """Compile PAW to parse questions and extract entities.
    
    Returns
    -------
    paw.function
        A callable PawFunction
    """
    # First compile to get the program object
    program = paw.compile(
        spec=(
            "Parse questions about NectarCAM camera runs. "
            "Extract the run number (e.g., '#635', '1234', or '1234-1235') and the field being asked about. "
            "Valid fields: trigger_mode, author, setup, light_source, date, modules, category, purpose. "
            "Return as JSON: {'run': '635', 'field': 'trigger_mode'}. "
            "If no run number is found, return {'run': null, 'field': null}."
        ),
        slug="nectarcam-question-parser",
        compiler="paw-4b-qwen3-0.6b",
        # compiler="paw-4b-gpt2",
    )
    
    # Store the program ID for cache management
    _PROGRAM_IDS["hybrid_parser"] = program.id
    
    # Now load the function
    parser = paw.function(program)
    return parser